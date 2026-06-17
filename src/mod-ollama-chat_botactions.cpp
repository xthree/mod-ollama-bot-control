#include "mod-ollama-chat_botactions.h"
#include "mod-ollama-chat_worldstate.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_handler.h"   // ChatChannelSourceLocal

#include "Playerbots.h"                // umbrella: AiObjectContext, PlayerbotAI, PlayerbotMgr, BOT_STATE_*
#include "Event.h"                     // Event (DoSpecificAction qualifier path)
#include "Player.h"
#include "Unit.h"
#include "ObjectGuid.h"
#include "ObjectAccessor.h"
#include "MotionMaster.h"
#include "WorldSession.h"
#include "WorldPacket.h"
#include "Opcodes.h"
#include "SharedDefines.h"
#include "Log.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <queue>
#include <sstream>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Per-bot opt-in. Read in the chat handler and written by the .ollama optin/out
// command; both run on the world thread, so no lock is required. In-memory for
// C3 (resets on restart); DB persistence is a later slice.
// ---------------------------------------------------------------------------
static std::unordered_map<uint64_t, bool> g_BotActionOptIn;

bool IsBotActionOptIn(Player* bot)
{
    if (!bot)
        return false;
    auto it = g_BotActionOptIn.find(bot->GetGUID().GetRawValue());
    return it != g_BotActionOptIn.end() && it->second;
}

void SetBotActionOptIn(Player* bot, bool optedIn)
{
    if (!bot)
        return;
    g_BotActionOptIn[bot->GetGUID().GetRawValue()] = optedIn;
}

bool SenderHasEngagedBot(Player* sender)
{
    if (!sender)
        return false;
    for (auto const& kv : g_BotActionOptIn)   // only opted-in bots — small set
    {
        if (!kv.second)
            continue;
        Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(kv.first));
        if (!bot)
            continue;
        PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (ai && ai->IsExternallyControlled() && ai->GetMaster() == sender)
            return true;
    }
    return false;
}

// --- Per-bot short-term action memory (world thread only) ---------------------
// A few recent actions so an ambiguous command ("ok this one now") is read in the
// context of what the bot has been doing ("we've been attacking wolves").
static std::unordered_map<uint64_t, std::deque<std::string>> g_BotRecentActions;

static void RecordBotAction(uint64_t botGuid, const std::string& desc)
{
    auto& dq = g_BotRecentActions[botGuid];
    dq.push_back(desc);
    while (dq.size() > 5)
        dq.pop_front();
}

static std::string RecentActionsText(uint64_t botGuid)
{
    auto it = g_BotRecentActions.find(botGuid);
    if (it == g_BotRecentActions.end() || it->second.empty())
        return "";
    std::string s;
    for (const std::string& d : it->second)
        s += "- " + d + "\n";
    return s;
}

// --- Conversation memory (shared with the chat path's g_BotConversationHistory) -
// Reusing the same store means an opted-in bot remembers what was said whether the
// message went through the chat path or the action path (so it can recall e.g. a
// name the player told it earlier).
static void RecordConversation(uint64_t botGuid, uint64_t playerGuid,
                               const std::string& msg, const std::string& reply)
{
    if (!g_EnableChatHistory || reply.empty())
        return;
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    auto& dq = g_BotConversationHistory[botGuid][playerGuid];
    dq.push_back({ msg, reply });
    while (g_MaxConversationHistory > 0 && dq.size() > g_MaxConversationHistory)
        dq.pop_front();
}

// --- Shared, attributed group conversation (world thread only) ----------------
// One transcript per bot covering EVERYONE who has spoken to it, each line tagged
// with the speaker. This is how a real player experiences a group chat: they hear
// everyone and remember who said what, so the bot can answer about a third person.
static std::unordered_map<uint64_t, std::deque<std::string>> g_BotSharedConvo;

static void RecordSharedConvo(uint64_t botGuid, const std::string& speaker, const std::string& text)
{
    if (text.empty())
        return;
    auto& dq = g_BotSharedConvo[botGuid];
    dq.push_back(speaker + ": " + text);
    while (dq.size() > 16)   // keep the last ~16 lines across all speakers
        dq.pop_front();
}

static std::string SharedConvoText(uint64_t botGuid)
{
    auto it = g_BotSharedConvo.find(botGuid);
    if (it == g_BotSharedConvo.end() || it->second.empty())
        return "";
    std::string s;
    for (const std::string& line : it->second)
        s += line + "\n";
    return s;
}

// --- Persistent pose holding (sit/sleep) --------------------------------------
// SetStandState alone doesn't stick: the bot's AI re-stands it next tick. We keep
// a per-bot target pose and re-apply it every world tick while the control lease
// is active; when the lease lapses (or the bot is released) we stop, and normal
// AI stands it back up.
static std::unordered_map<uint64_t, uint8_t> g_BotPosedState;   // botGuid -> UNIT_STAND_STATE_*

static void SetPose(uint64_t botGuid, uint8_t state) { g_BotPosedState[botGuid] = state; }
static void ClearPose(uint64_t botGuid)              { g_BotPosedState.erase(botGuid); }

static void MaintainPoses()
{
    for (auto it = g_BotPosedState.begin(); it != g_BotPosedState.end(); )
    {
        Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(it->first));
        PlayerbotAI* botAI = bot ? PlayerbotsMgr::instance().GetPlayerbotAI(bot) : nullptr;
        if (!bot || !botAI || !botAI->IsExternallyControlled())
        {
            it = g_BotPosedState.erase(it);     // released -> let it stand on its own
            continue;
        }
        if (bot->getStandState() != it->second)
            bot->SetStandState(it->second);
        ++it;
    }
}

// ---------------------------------------------------------------------------
// Worker-thread -> world-thread action queue.
// ---------------------------------------------------------------------------
namespace
{
    struct PendingBotAction
    {
        uint64_t botGuid;
        uint64_t senderGuid;
        int sourceLocal;
        std::string say;
        std::string message;
        BotActionCommand cmd;
    };

    // Recover an emote name from the player's message when the model omitted it.
    std::string EmoteFromMessage(const std::string& message)
    {
        std::string m = message;
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        static const char* kEmotes[] = { "dismount", "sit", "sleep", "lay down", "lie down", "stand up", "stand",
                                         "dance", "cheer", "wave", "laugh", "bow", "roar", "salute",
                                         "clap", "applaud", "cry", "flex", "kneel", "point", "rude",
                                         "train", "wink", "thank", "hug", "victory", "flirt", "chicken" };
        for (const char* e : kEmotes)
            if (m.find(e) != std::string::npos)
            {
                std::string r = e;
                if (r == "lay down" || r == "lie down") return "sleep";
                if (r == "stand up") return "stand";
                return r;
            }
        return "";
    }

    std::mutex g_BotActionQueueMutex;
    std::queue<PendingBotAction> g_BotActionQueue;

    bool ActionAllowed(const std::string& type)
    {
        if (type.empty() || type == "none")
            return false;
        if (g_AllowedActions.empty())
            return true;  // unset == allow the full v1 set
        return std::find(g_AllowedActions.begin(), g_AllowedActions.end(), type) != g_AllowedActions.end();
    }
}

void EnqueueBotAction(uint64_t botGuid, uint64_t senderGuid, int sourceLocal,
                      const std::string& say, const std::string& message, const BotActionCommand& cmd)
{
    std::lock_guard<std::mutex> lock(g_BotActionQueueMutex);
    g_BotActionQueue.push(PendingBotAction{ botGuid, senderGuid, sourceLocal, say, message, cmd });
}

// ---------------------------------------------------------------------------
// Prompt building (world thread; reads nearby units).
// ---------------------------------------------------------------------------
std::string BuildBotActionPrompt(Player* bot, Player* sender, const std::string& message)
{
    std::string worldState = BuildBotWorldStateContext(bot);
    std::string botName = bot ? bot->GetName() : "bot";
    std::string senderName = sender ? sender->GetName() : "someone";

    float px = sender ? sender->GetPositionX() : 0.0f;
    float py = sender ? sender->GetPositionY() : 0.0f;
    float pz = sender ? sender->GetPositionZ() : 0.0f;

    uint64_t botGuid = bot ? bot->GetGUID().GetRawValue() : 0;
    uint64_t senderGuid = sender ? sender->GetGUID().GetRawValue() : 0;
    std::string recent = RecentActionsText(botGuid);
    std::string convo = SharedConvoText(botGuid);
    (void)senderGuid;

    std::ostringstream p;
    p << "You are " << botName << ", a friendly companion adventuring alongside the players. "
      << "Act like a real player — chat naturally, with personality, and remember what people tell "
      << "you — but you are loyal and you DO what your companions ask of you.\n";
    if (!convo.empty())
        p << "The conversation around you so far (each line is tagged with who said it — "
          << "remember who said what, including OTHER people besides " << senderName
          << ", and answer questions about them accurately):\n" << convo << "\n";
    if (!recent.empty())
        p << "What you have been doing recently (use this to read short or vague requests "
          << "like \"this one now\" or \"get it\" in context):\n" << recent << "\n";
    p << senderName << " just said to you: \"" << message << "\".\n"
      << "Nearby units (use the exact guid number shown; do not invent one):\n"
      << worldState
      << senderName << " is standing at x=" << px << " y=" << py << " z=" << pz << ".\n\n"
      << "Decide what to do. When " << senderName << " gives you a command or asks you to DO "
      << "something (attack/kill, follow, come, stay, sit, move, an emote), you ALWAYS carry it out — "
      << "set that exact action. NEVER refuse, argue, or just talk about it; you may be playful in your "
      << "\"say\", but you still do it. If it is only conversation, banter, or a question, use "
      << "action=\"none\" and reply.\n"
      << "- attack a unit: action=\"attack\", guid=<that unit's guid number>.\n"
      << "- follow " << senderName << " around: action=\"follow\".\n"
      << "- come over to " << senderName << " (walk up and stop near them, NOT follow): action=\"come\".\n"
      << "- stop and hold your current spot: action=\"stay\".\n"
      << "- go to a specific spot: action=\"moveto\" with x,y,z.\n"
      << "- a gesture or pose, ONLY if they explicitly ask you to do that gesture by name "
      << "(e.g. \"sit\", \"dance\", \"bow\", \"wave\"): action=\"emote\" and emote=<that name> "
      << "(any standard WoW emote, plus sit, sleep, stand, dismount: dance, bow, wave, rude, train, "
      << "wink, kneel, cheer, laugh, salute, flex, point, thank, hug, roar, clap, chicken, victory). "
      << "Do NOT emote just because you are chatting about something.\n"
      << "Always include a natural, in-character \"say\" of ONE or TWO short sentences — grounded and "
      << "on-topic, like a real person in chat. Do not ramble, repeat yourself, or invent absurd "
      << "scenarios; vary your wording and don't start every reply the same way.";
    return p.str();
}

std::string BotActionSchema()
{
    // Flat, easy-for-a-small-model schema. Ollama constrains output to this.
    return R"JSON({
      "type":"object",
      "properties":{
        "say":{"type":"string","maxLength":200},
        "action":{"type":"string","enum":["attack","follow","come","stay","moveto","emote","none"]},
        "guid":{"type":"integer"},
        "x":{"type":"number"},
        "y":{"type":"number"},
        "z":{"type":"number"},
        "emote":{"type":"string"}
      },
      "required":["say","action"]
    })JSON";
}

// ---------------------------------------------------------------------------
// Response parsing: extract the first balanced {...} object, then read fields.
// ---------------------------------------------------------------------------
bool ParseBotActionResponse(const std::string& response, std::string& sayOut, BotActionCommand& cmdOut)
{
    size_t start = response.find('{');
    if (start == std::string::npos)
        return false;

    int depth = 0;
    size_t end = std::string::npos;
    bool inStr = false, esc = false;
    for (size_t i = start; i < response.size(); ++i)
    {
        char c = response[i];
        if (inStr)
        {
            if (esc)            esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"')  inStr = false;
            continue;
        }
        if (c == '"')      { inStr = true; }
        else if (c == '{') { depth++; }
        else if (c == '}') { if (--depth == 0) { end = i; break; } }
    }
    if (end == std::string::npos)
        return false;

    std::string jsonStr = response.substr(start, end - start + 1);

    try
    {
        nlohmann::json j = nlohmann::json::parse(jsonStr);

        // Output is schema-constrained to a flat {say, action, guid?, x,y,z?, emote?}.
        if (j.contains("say") && j["say"].is_string())
            sayOut = j["say"].get<std::string>();

        if (j.contains("action") && j["action"].is_string())
            cmdOut.type = j["action"].get<std::string>();

        std::transform(cmdOut.type.begin(), cmdOut.type.end(), cmdOut.type.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        if (j.contains("guid"))
        {
            auto& g = j["guid"];
            if (g.is_number_unsigned())     cmdOut.targetGuid = g.get<uint64_t>();
            else if (g.is_number_integer()) cmdOut.targetGuid = static_cast<uint64_t>(g.get<int64_t>());
            else if (g.is_string())         cmdOut.targetGuid = std::strtoull(g.get<std::string>().c_str(), nullptr, 10);
        }
        if (j.contains("x") && j.contains("y") && j.contains("z") &&
            j["x"].is_number() && j["y"].is_number() && j["z"].is_number())
        {
            cmdOut.hasPos = true;
            cmdOut.x = j["x"].get<float>();
            cmdOut.y = j["y"].get<float>();
            cmdOut.z = j["z"].get<float>();
        }
        if (j.contains("emote") && j["emote"].is_string())
            cmdOut.param = j["emote"].get<std::string>();

        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

// ---------------------------------------------------------------------------
// World-thread execution.
// ---------------------------------------------------------------------------
namespace
{
    void SpeakReply(PlayerbotAI* botAI, Player* sender, int sourceLocal, const std::string& say)
    {
        if (say.empty())
            return;

        switch (sourceLocal)
        {
            case SRC_WHISPER_LOCAL:
                if (sender) botAI->Whisper(say, sender->GetName());
                else        botAI->Say(say);
                break;
            case SRC_PARTY_LOCAL:  botAI->SayToParty(say); break;
            case SRC_RAID_LOCAL:   botAI->SayToRaid(say);  break;
            case SRC_GUILD_LOCAL:
            case SRC_OFFICER_LOCAL:botAI->SayToGuild(say); break;
            case SRC_YELL_LOCAL:   botAI->Yell(say);       break;
            default:               botAI->Say(say);        break;
        }
    }

    bool ValidAttackTarget(Player* bot, Unit* u)
    {
        // Permissive on purpose: any living unit in the world, so the bot can be told to
        // attack anything (critters included). bot->Attack enforces whatever the core allows.
        (void)bot;
        return u && u->IsInWorld() && !u->isDead();
    }

    // Resolve who to attack. Replicates AttackAction::Attack for an arbitrary unit
    // (the "attack my target" action only reads the master's selection). The model
    // often omits the guid, so fall back to (a) the commanding player's current
    // target ("help me" while you fight something), then (b) the nearest hostile.
    void ExecuteAttack(Player* bot, PlayerbotAI* botAI, Player* sender, uint64_t targetGuidRaw)
    {
        Unit* target = targetGuidRaw ? botAI->GetUnit(ObjectGuid(targetGuidRaw)) : nullptr;

        if (!ValidAttackTarget(bot, target) && sender && sender->GetTarget())
        {
            Unit* u = botAI->GetUnit(sender->GetTarget());
            if (ValidAttackTarget(bot, u))
                target = u;
        }

        if (!ValidAttackTarget(bot, target))
        {
            Unit* nearest = nullptr;
            float best = 99999.0f;
            GuidVector hostiles = botAI->GetAiObjectContext()->GetValue<GuidVector>("nearest hostile npcs")->Get();
            for (ObjectGuid const& g : hostiles)
            {
                Unit* u = botAI->GetUnit(g);
                if (!ValidAttackTarget(bot, u))
                    continue;
                float d = bot->GetDistance(u);
                if (d < best) { best = d; nearest = u; }
            }
            target = nearest;
        }

        if (!ValidAttackTarget(bot, target))
            return;   // nothing valid to attack

        bot->SetSelection(target->GetGUID());
        botAI->GetAiObjectContext()->GetValue<Unit*>("current target")->Set(target);
        botAI->GetAiObjectContext()->GetValue<GuidVector>("prioritized targets")->Set({ target->GetGUID() });
        botAI->ChangeEngine(BOT_STATE_COMBAT);
        bot->Attack(target, bot->IsWithinMeleeRange(target));
        RecordBotAction(bot->GetGUID().GetRawValue(), "attacked " + target->GetName());
    }

    // Perform any standard WoW emote by name: persistent poses (sit/sleep/stand)
    // via stand-state, everything else as a real /emote (animation + social text).
    void ExecuteEmote(Player* bot, PlayerbotAI* botAI, const std::string& rawName, const std::string& message)
    {
        std::string name = rawName.empty() ? EmoteFromMessage(message) : rawName;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        while (!name.empty() && (name.front() == '/' || name.front() == ' ')) name.erase(name.begin());
        while (!name.empty() && name.back() == ' ') name.pop_back();
        if (name.empty())
            return;   // no resolvable emote -> do nothing (don't randomly wave)

        if (name == "dismount" || name == "unmount")                             { bot->Dismount(); return; }

        // Persistent stand-state poses.
        if (name == "sit" || name == "sitdown")                                  { bot->SetStandState(UNIT_STAND_STATE_SIT);   return; }
        if (name == "sleep" || name == "lay" || name == "laydown" || name == "liedown") { bot->SetStandState(UNIT_STAND_STATE_SLEEP); return; }
        if (name == "stand" || name == "standup" || name == "getup")             { bot->SetStandState(UNIT_STAND_STATE_STAND); return; }

        static const std::unordered_map<std::string, uint32_t> kTextEmotes = {
            {"agree",TEXT_EMOTE_AGREE},{"amaze",TEXT_EMOTE_AMAZE},{"angry",TEXT_EMOTE_ANGRY},
            {"apologize",TEXT_EMOTE_APOLOGIZE},{"sorry",TEXT_EMOTE_APOLOGIZE},{"applaud",TEXT_EMOTE_APPLAUD},
            {"bashful",TEXT_EMOTE_BASHFUL},{"beckon",TEXT_EMOTE_BECKON},{"beg",TEXT_EMOTE_BEG},
            {"bite",TEXT_EMOTE_BITE},{"blink",TEXT_EMOTE_BLINK},{"blush",TEXT_EMOTE_BLUSH},
            {"bonk",TEXT_EMOTE_BONK},{"bored",TEXT_EMOTE_BORED},{"bounce",TEXT_EMOTE_BOUNCE},
            {"bow",TEXT_EMOTE_BOW},{"brb",TEXT_EMOTE_BRB},{"burp",TEXT_EMOTE_BURP},{"bye",TEXT_EMOTE_BYE},
            {"cackle",TEXT_EMOTE_CACKLE},{"cheer",TEXT_EMOTE_CHEER},{"chicken",TEXT_EMOTE_CHICKEN},
            {"chuckle",TEXT_EMOTE_CHUCKLE},{"clap",TEXT_EMOTE_CLAP},{"confused",TEXT_EMOTE_CONFUSED},
            {"congratulate",TEXT_EMOTE_CONGRATULATE},{"grats",TEXT_EMOTE_CONGRATULATE},{"cough",TEXT_EMOTE_COUGH},
            {"cower",TEXT_EMOTE_COWER},{"cringe",TEXT_EMOTE_CRINGE},{"cry",TEXT_EMOTE_CRY},
            {"curious",TEXT_EMOTE_CURIOUS},{"curtsey",TEXT_EMOTE_CURTSEY},{"dance",TEXT_EMOTE_DANCE},
            {"drink",TEXT_EMOTE_DRINK},{"drool",TEXT_EMOTE_DROOL},{"eat",TEXT_EMOTE_EAT},
            {"flex",TEXT_EMOTE_FLEX},{"flirt",TEXT_EMOTE_FLIRT},{"frown",TEXT_EMOTE_FROWN},
            {"gasp",TEXT_EMOTE_GASP},{"giggle",TEXT_EMOTE_GIGGLE},{"glare",TEXT_EMOTE_GLARE},
            {"gloat",TEXT_EMOTE_GLOAT},{"greet",TEXT_EMOTE_GREET},{"grin",TEXT_EMOTE_GRIN},
            {"groan",TEXT_EMOTE_GROAN},{"grovel",TEXT_EMOTE_GROVEL},{"growl",TEXT_EMOTE_GROWL},
            {"guffaw",TEXT_EMOTE_GUFFAW},{"hail",TEXT_EMOTE_HAIL},{"happy",TEXT_EMOTE_HAPPY},
            {"hello",TEXT_EMOTE_HELLO},{"hi",TEXT_EMOTE_HELLO},{"hug",TEXT_EMOTE_HUG},
            {"hungry",TEXT_EMOTE_HUNGRY},{"insult",TEXT_EMOTE_INSULT},{"kiss",TEXT_EMOTE_KISS},
            {"kneel",TEXT_EMOTE_KNEEL},{"laugh",TEXT_EMOTE_LAUGH},{"lick",TEXT_EMOTE_LICK},
            {"listen",TEXT_EMOTE_LISTEN},{"look",TEXT_EMOTE_LOOK},{"lost",TEXT_EMOTE_LOST},
            {"love",TEXT_EMOTE_LOVE},{"moan",TEXT_EMOTE_MOAN},{"mock",TEXT_EMOTE_MOCK},
            {"moo",TEXT_EMOTE_MOO},{"moon",TEXT_EMOTE_MOON},{"mourn",TEXT_EMOTE_MOURN},
            {"no",TEXT_EMOTE_NO},{"nod",TEXT_EMOTE_NOD},{"yes",TEXT_EMOTE_NOD},{"panic",TEXT_EMOTE_PANIC},
            {"peer",TEXT_EMOTE_PEER},{"pet",TEXT_EMOTE_PET},{"pinch",TEXT_EMOTE_PINCH},
            {"plead",TEXT_EMOTE_PLEAD},{"point",TEXT_EMOTE_POINT},{"poke",TEXT_EMOTE_POKE},
            {"ponder",TEXT_EMOTE_PONDER},{"pounce",TEXT_EMOTE_POUNCE},{"pout",TEXT_EMOTE_POUT},
            {"praise",TEXT_EMOTE_PRAISE},{"pray",TEXT_EMOTE_PRAY},{"promise",TEXT_EMOTE_PROMISE},
            {"proud",TEXT_EMOTE_PROUD},{"punch",TEXT_EMOTE_PUNCH},{"purr",TEXT_EMOTE_PURR},
            {"puzzle",TEXT_EMOTE_PUZZLE},{"raise",TEXT_EMOTE_RAISE},{"ready",TEXT_EMOTE_READY},
            {"roar",TEXT_EMOTE_ROAR},{"rofl",TEXT_EMOTE_ROFL},{"lol",TEXT_EMOTE_ROFL},
            {"rolleyes",TEXT_EMOTE_ROLLEYES},{"rude",TEXT_EMOTE_RUDE},{"ruffle",TEXT_EMOTE_RUFFLE},
            {"sad",TEXT_EMOTE_SAD},{"salute",TEXT_EMOTE_SALUTE},{"scared",TEXT_EMOTE_SCARED},
            {"scoff",TEXT_EMOTE_SCOFF},{"scold",TEXT_EMOTE_SCOLD},{"scowl",TEXT_EMOTE_SCOWL},
            {"scratch",TEXT_EMOTE_SCRATCH},{"search",TEXT_EMOTE_SEARCH},{"serious",TEXT_EMOTE_SERIOUS},
            {"sexy",TEXT_EMOTE_SEXY},{"shake",TEXT_EMOTE_SHAKE},{"shimmy",TEXT_EMOTE_SHIMMY},
            {"shiver",TEXT_EMOTE_SHIVER},{"shoo",TEXT_EMOTE_SHOO},{"shout",TEXT_EMOTE_SHOUT},
            {"shrug",TEXT_EMOTE_SHRUG},{"shudder",TEXT_EMOTE_SHUDDER},{"shy",TEXT_EMOTE_SHY},
            {"sigh",TEXT_EMOTE_SIGH},{"sing",TEXT_EMOTE_SING},{"slap",TEXT_EMOTE_SLAP},
            {"smack",TEXT_EMOTE_SMACK},{"smile",TEXT_EMOTE_SMILE},{"smirk",TEXT_EMOTE_SMIRK},
            {"snap",TEXT_EMOTE_SNAP},{"snarl",TEXT_EMOTE_SNARL},{"sneak",TEXT_EMOTE_SNEAK},
            {"sneeze",TEXT_EMOTE_SNEEZE},{"snicker",TEXT_EMOTE_SNICKER},{"sniff",TEXT_EMOTE_SNIFF},
            {"snort",TEXT_EMOTE_SNORT},{"snub",TEXT_EMOTE_SNUB},{"soothe",TEXT_EMOTE_SOOTHE},
            {"spit",TEXT_EMOTE_SPIT},{"squeal",TEXT_EMOTE_SQUEAL},{"stare",TEXT_EMOTE_STARE},
            {"stink",TEXT_EMOTE_STINK},{"surprised",TEXT_EMOTE_SURPRISED},{"surrender",TEXT_EMOTE_SURRENDER},
            {"suspicious",TEXT_EMOTE_SUSPICIOUS},{"sweat",TEXT_EMOTE_SWEAT},{"talk",TEXT_EMOTE_TALK},
            {"tap",TEXT_EMOTE_TAP},{"taunt",TEXT_EMOTE_TAUNT},{"tease",TEXT_EMOTE_TEASE},
            {"thank",TEXT_EMOTE_THANK},{"thanks",TEXT_EMOTE_THANK},{"think",TEXT_EMOTE_THINK},
            {"thirsty",TEXT_EMOTE_THIRSTY},{"threaten",TEXT_EMOTE_THREATEN},{"tickle",TEXT_EMOTE_TICKLE},
            {"tired",TEXT_EMOTE_TIRED},{"toast",TEXT_EMOTE_TOAST},{"train",TEXT_EMOTE_TRAIN},
            {"twiddle",TEXT_EMOTE_TWIDDLE},{"victory",TEXT_EMOTE_VICTORY},{"violin",TEXT_EMOTE_VIOLIN},
            {"wave",TEXT_EMOTE_WAVE},{"welcome",TEXT_EMOTE_WELCOME},{"whine",TEXT_EMOTE_WHINE},
            {"whistle",TEXT_EMOTE_WHISTLE},{"wink",TEXT_EMOTE_WINK},{"work",TEXT_EMOTE_WORK},
            {"yawn",TEXT_EMOTE_YAWN},
        };

        auto it = kTextEmotes.find(name);
        if (it != kTextEmotes.end())
        {
            WorldPacket data(CMSG_TEXT_EMOTE, 4 + 4 + 8);
            data << uint32_t(it->second);
            data << uint32_t(0);
            data << ObjectGuid::Empty;
            bot->GetSession()->HandleTextEmoteOpcode(data);
            return;
        }

        // Unknown name: fall back to the engine's animation map.
        botAI->DoSpecificAction("emote", Event(), false, name);
    }

    void ExecuteBotAction(const PendingBotAction& pa)
    {
        Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(pa.botGuid));
        if (!bot)
            return;
        PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!botAI)
            return;

        Player* sender = ObjectAccessor::FindPlayer(ObjectGuid(pa.senderGuid));

        const BotActionCommand& cmd = pa.cmd;
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[OllamaBotControl] drain: bot {} type='{}' guid={} say='{}'",
                     bot->GetName(), cmd.type, cmd.targetGuid, pa.say);

        // Speak the reply first (the bot answers in natural speech AND acts).
        SpeakReply(botAI, sender, pa.sourceLocal, pa.say);

        // Remember this exchange, per-character, so the bot recalls what THIS player
        // told it (shared with the chat path's history store).
        RecordConversation(pa.botGuid, pa.senderGuid, pa.message, pa.say);

        // And into the shared, attributed group transcript so the bot is aware of
        // everyone in the conversation (e.g. can answer about a third person).
        RecordSharedConvo(pa.botGuid, sender ? sender->GetName() : "Someone", pa.message);
        RecordSharedConvo(pa.botGuid, bot->GetName(), pa.say);

        // Engagement: being addressed pauses the bot's autonomous questing so it stops
        // and attends to you naturally — no explicit "stay" needed. We set the master to
        // whoever is talking so the handler keeps routing their later unaddressed messages
        // to this bot (no need to repeat its name), and renew the control lease. The strip
        // removes only the autonomous movers, leaving an active follow/stay/pose intact.
        if (sender)
            botAI->SetMaster(sender);
        botAI->SetExternalControl(g_ControlDurationSeconds);
        botAI->ChangeStrategy("-grind,-new rpg,-rpg,-move random,-travel,-lfg,-bg,-start duel",
                              BOT_STATE_NON_COMBAT);

        if (!ActionAllowed(cmd.type))
        {
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[OllamaBotControl] bot {} engaged (conversation), no action", bot->GetName());
            return;   // none: engaged + chatting, no physical action
        }

        // A movement/combat command means stop holding a sit/sleep pose.
        if (cmd.type == "attack" || cmd.type == "follow" || cmd.type == "moveto" || cmd.type == "come")
            ClearPose(pa.botGuid);

        if (cmd.type == "attack")
        {
            ExecuteAttack(bot, botAI, sender, cmd.targetGuid);
        }
        else if (cmd.type == "follow")
        {
            if (sender)
                botAI->SetMaster(sender);
            botAI->DoSpecificAction("follow chat shortcut");
            RecordBotAction(bot->GetGUID().GetRawValue(),
                            "started following " + (sender ? sender->GetName() : std::string("the player")));
        }
        else if (cmd.type == "moveto")
        {
            float x = cmd.x, y = cmd.y, z = cmd.z;
            if (!cmd.hasPos && sender)
            {
                x = sender->GetPositionX();
                y = sender->GetPositionY();
                z = sender->GetPositionZ();
            }
            if (x != 0.0f || y != 0.0f || z != 0.0f)
            {
                bot->GetMotionMaster()->Clear();
                bot->GetMotionMaster()->MovePoint(0, x, y, z);
                RecordBotAction(bot->GetGUID().GetRawValue(), "moved to a location");
            }
        }
        else if (cmd.type == "come")
        {
            // Walk up to the player but stop a couple of yards short (not on top of them),
            // and just hold there afterward (this is NOT a follow).
            if (sender)
            {
                float px = sender->GetPositionX(), py = sender->GetPositionY(), pz = sender->GetPositionZ();
                float dx = bot->GetPositionX() - px, dy = bot->GetPositionY() - py;
                float len = std::sqrt(dx * dx + dy * dy);
                float stop = 2.5f;
                float tx = px, ty = py;
                if (len > 0.1f) { tx = px + dx / len * stop; ty = py + dy / len * stop; }
                botAI->ChangeStrategy("-follow", BOT_STATE_NON_COMBAT);
                bot->GetMotionMaster()->Clear();
                bot->GetMotionMaster()->MovePoint(0, tx, ty, pz);
                RecordBotAction(bot->GetGUID().GetRawValue(), "came over to " + sender->GetName());
            }
        }
        else if (cmd.type == "stay")
        {
            // Hold the current position (not a follow). +stay keeps the bot put.
            botAI->ChangeStrategy("-follow,+stay", BOT_STATE_NON_COMBAT);
            bot->GetMotionMaster()->Clear();
            bot->StopMoving();
            RecordBotAction(bot->GetGUID().GetRawValue(), "stayed put");
        }
        else if (cmd.type == "emote")
        {
            std::string name = cmd.param.empty() ? EmoteFromMessage(pa.message) : cmd.param;
            std::string elow = name;
            std::transform(elow.begin(), elow.end(), elow.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            if (elow.empty())
            {
                // Model picked emote but gave no resolvable name -> just the spoken reply,
                // no random wave.
            }
            else if (elow == "sit" || elow == "sitdown" || elow == "sleep" ||
                     elow == "lay" || elow == "laydown" || elow == "liedown")
            {
                // Persistent pose: lease + hold it (MaintainPoses re-applies each tick),
                // stop moving, and remember the target stand-state.
                botAI->SetExternalControl(g_ControlDurationSeconds);
                botAI->ChangeStrategy("-follow,-grind,-new rpg,-rpg,-move random,-travel,-lfg,-bg,-start duel,+stay",
                                      BOT_STATE_NON_COMBAT);
                bot->GetMotionMaster()->Clear();
                bot->StopMoving();
                SetPose(pa.botGuid, (elow == "sit" || elow == "sitdown") ? UNIT_STAND_STATE_SIT
                                                                         : UNIT_STAND_STATE_SLEEP);
                ExecuteEmote(bot, botAI, name, pa.message);
                RecordBotAction(pa.botGuid, "is " + std::string(elow.substr(0,3) == "sit" ? "sitting" : "lying down"));
            }
            else
            {
                if (elow == "stand" || elow == "standup" || elow == "getup")
                    ClearPose(pa.botGuid);   // getting up cancels a held pose
                ExecuteEmote(bot, botAI, name, pa.message);
                RecordBotAction(pa.botGuid, "performed the " + elow + " emote");
            }
        }

        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[OllamaBotControl] Bot {} executed action '{}'", bot->GetName(), cmd.type);
    }
}  // namespace

void DrainBotActionQueue()
{
    std::queue<PendingBotAction> local;
    {
        std::lock_guard<std::mutex> lock(g_BotActionQueueMutex);
        std::swap(local, g_BotActionQueue);
    }

    while (!local.empty())
    {
        ExecuteBotAction(local.front());
        local.pop();
    }

    // Hold any sit/sleep poses against the bot's AI standing them back up.
    MaintainPoses();
}
