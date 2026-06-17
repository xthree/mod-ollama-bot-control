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
#include "Log.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
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
static std::string ConversationContext(uint64_t botGuid, uint64_t playerGuid, const std::string& playerName)
{
    if (!g_EnableChatHistory)
        return "";
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return "";
    auto pIt = botIt->second.find(playerGuid);
    if (pIt == botIt->second.end() || pIt->second.empty())
        return "";
    std::string s;
    for (const auto& e : pIt->second)
    {
        s += playerName + ": " + e.first + "\n";
        s += "You: " + e.second + "\n";
    }
    return s;
}

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
        static const char* kEmotes[] = { "dance", "cheer", "wave", "laugh", "bow", "roar",
                                         "salute", "clap", "applaud", "cry", "flex", "kneel", "point", "sleep" };
        for (const char* e : kEmotes)
            if (m.find(e) != std::string::npos)
                return e;
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
    std::string convo = ConversationContext(botGuid, senderGuid, senderName);

    std::ostringstream p;
    p << "You are " << botName << ", a character in World of Warcraft. Act like a real player: "
      << "chat naturally, have a personality, and remember what people tell you.\n";
    if (!convo.empty())
        p << "Your earlier conversation with " << senderName
          << " (this is what THIS person has said to you before — remember it):\n" << convo << "\n";
    if (!recent.empty())
        p << "What you have been doing recently (use this to read short or vague requests "
          << "like \"this one now\" or \"get it\" in context):\n" << recent << "\n";
    p << senderName << " just said to you: \"" << message << "\".\n"
      << "Nearby units (use the exact guid number shown; do not invent one):\n"
      << worldState
      << senderName << " is standing at x=" << px << " y=" << py << " z=" << pz << ".\n\n"
      << "MOST messages are just conversation, banter, or questions — for those set action=\"none\" "
      << "and simply reply in \"say\" like a person would (answer questions, react, joke). "
      << "ONLY choose a physical action when " << senderName << " clearly asks for that specific thing right now:\n"
      << "- attack a unit: action=\"attack\", guid=<that unit's guid number>.\n"
      << "- follow " << senderName << ": action=\"follow\".\n"
      << "- move somewhere: action=\"moveto\" with x,y,z (to come to " << senderName << " use their x,y,z above).\n"
      << "- a gesture: action=\"emote\", emote=one of dance, cheer, wave, laugh, bow, roar.\n"
      << "When in doubt, use action=\"none\" and just talk. Always include a natural, in-character \"say\" "
      << "of ONE or TWO short sentences — speak like a real person in chat, do not ramble.";
    return p.str();
}

std::string BotActionSchema()
{
    // Flat, easy-for-a-small-model schema. Ollama constrains output to this.
    return R"JSON({
      "type":"object",
      "properties":{
        "say":{"type":"string","maxLength":200},
        "action":{"type":"string","enum":["attack","follow","moveto","emote","none"]},
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
        return u && u->IsInWorld() && !u->isDead() && bot->IsValidAttackTarget(u);
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

        if (!ActionAllowed(cmd.type))
        {
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[OllamaBotControl] action '{}' not allowed/none -> skipped", cmd.type);
            return;
        }

        // Take a time-boxed control lease for state/strategy-based actions so the
        // engine doesn't yank the bot back to autonomy before it acts. Emotes are
        // instantaneous and need no lease. The lease auto-expires (renewed per
        // command), so the bot resumes its own life once you stop interacting.
        if (cmd.type == "attack" || cmd.type == "follow" || cmd.type == "moveto")
        {
            botAI->SetExternalControl(g_ControlDurationSeconds);
            // Strip the random bot's autonomous movement/questing strategies so they
            // don't fight the commanded action (e.g. "move random"/"new rpg" vs follow,
            // which makes the bot jitter). They are re-added by ResetStrategies once the
            // lease lapses, restoring normal autonomy.
            botAI->ChangeStrategy("-grind,-new rpg,-rpg,-move random,-travel,-lfg,-bg,-start duel",
                                  BOT_STATE_NON_COMBAT);
        }

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
        else if (cmd.type == "emote")
        {
            std::string name = cmd.param;
            if (name.empty())
                name = EmoteFromMessage(pa.message);   // model omitted the emote field
            if (name.empty())
                name = "wave";
            botAI->DoSpecificAction("emote", Event(), false, name);
            RecordBotAction(bot->GetGUID().GetRawValue(), "performed the " + name + " emote");
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
}
