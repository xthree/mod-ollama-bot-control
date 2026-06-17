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

    std::ostringstream p;
    p << "You are " << botName << ", a World of Warcraft character. "
      << senderName << ", your companion, said to you: \"" << message << "\".\n"
      << "Nearby units (use the exact guid number shown; do not invent one):\n"
      << worldState
      << senderName << " is standing at x=" << px << " y=" << py << " z=" << pz << ".\n\n"
      << "Choose ONE action in response and fill the matching field:\n"
      << "- To fight a unit: action=\"attack\" and guid=<that unit's guid number>.\n"
      << "- To follow " << senderName << ": action=\"follow\".\n"
      << "- To move to a spot: action=\"moveto\" and x,y,z. To come to " << senderName
      << ", use their x,y,z above.\n"
      << "- To perform a gesture: action=\"emote\" and emote=one of dance, cheer, wave, laugh, bow, roar.\n"
      << "- If nothing should be done: action=\"none\".\n"
      << "Always include a short, in-character spoken reply in \"say\". Do what "
      << senderName << " asks of you.";
    return p.str();
}

std::string BotActionSchema()
{
    // Flat, easy-for-a-small-model schema. Ollama constrains output to this.
    return R"JSON({
      "type":"object",
      "properties":{
        "say":{"type":"string"},
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

    // Replicate AttackAction::Attack for an arbitrary GUID (the "attack my target"
    // action only reads the master's selection, so it cannot target a chosen guid).
    void ExecuteAttack(Player* bot, PlayerbotAI* botAI, uint64_t targetGuidRaw)
    {
        if (!targetGuidRaw)
            return;

        ObjectGuid guid(targetGuidRaw);
        Unit* target = botAI->GetUnit(guid);

        // GUID re-validation: reject hallucinated / invalid targets silently.
        if (!target || !target->IsInWorld() || target->isDead())
            return;
        if (!bot->IsValidAttackTarget(target))
            return;
        if (!bot->IsWithinLOSInMap(target))
            return;

        bot->SetSelection(guid);
        botAI->GetAiObjectContext()->GetValue<Unit*>("current target")->Set(target);
        botAI->GetAiObjectContext()->GetValue<GuidVector>("prioritized targets")->Set({ guid });
        botAI->ChangeEngine(BOT_STATE_COMBAT);
        bot->Attack(target, bot->IsWithinMeleeRange(target));
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
            botAI->SetExternalControl(g_ControlDurationSeconds);

        if (cmd.type == "attack")
        {
            ExecuteAttack(bot, botAI, cmd.targetGuid);
        }
        else if (cmd.type == "follow")
        {
            if (sender)
                botAI->SetMaster(sender);
            botAI->DoSpecificAction("follow chat shortcut");
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
