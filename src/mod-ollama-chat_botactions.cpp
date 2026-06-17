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
        BotActionCommand cmd;
    };

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
                      const std::string& say, const BotActionCommand& cmd)
{
    std::lock_guard<std::mutex> lock(g_BotActionQueueMutex);
    g_BotActionQueue.push(PendingBotAction{ botGuid, senderGuid, sourceLocal, say, cmd });
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
    p << "You are " << botName << ", a World of Warcraft character controlled as a bot. "
      << "A player named " << senderName << " said to you: \"" << message << "\".\n"
      << "Nearby units (use ONLY these exact guid numbers; never invent a guid):\n"
      << worldState << "\n"
      << "The player " << senderName << " is at position x=" << px << " y=" << py << " z=" << pz << ".\n"
      << "Decide a single action to take in response. Reply with ONLY one JSON object, no other text:\n"
      << "{\"say\":\"<short in-character reply>\",\"command\":{\"type\":\"<attack|follow|moveto|emote|none>\",\"params\":{}}}\n"
      << "Params by command type:\n"
      << "- attack: {\"guid\": <a guid number from the list above>}\n"
      << "- follow: {} (you will follow " << senderName << ")\n"
      << "- moveto: {\"x\":<num>,\"y\":<num>,\"z\":<num>} (to go to " << senderName << ", use their position above)\n"
      << "- emote: {\"name\":\"<dance|cheer|wave|laugh|bow|roar>\"}\n"
      << "- none: {} if no physical action is appropriate.\n";
    return p.str();
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

        if (j.contains("say") && j["say"].is_string())
            sayOut = j["say"].get<std::string>();

        if (j.contains("command") && j["command"].is_object())
        {
            auto& c = j["command"];
            if (c.contains("type") && c["type"].is_string())
                cmdOut.type = c["type"].get<std::string>();

            std::transform(cmdOut.type.begin(), cmdOut.type.end(), cmdOut.type.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

            if (c.contains("params") && c["params"].is_object())
            {
                auto& pr = c["params"];
                if (pr.contains("guid"))
                {
                    if (pr["guid"].is_number_unsigned())     cmdOut.targetGuid = pr["guid"].get<uint64_t>();
                    else if (pr["guid"].is_number_integer()) cmdOut.targetGuid = static_cast<uint64_t>(pr["guid"].get<int64_t>());
                    else if (pr["guid"].is_string())         cmdOut.targetGuid = std::strtoull(pr["guid"].get<std::string>().c_str(), nullptr, 10);
                }
                if (pr.contains("x") && pr.contains("y") && pr.contains("z") &&
                    pr["x"].is_number() && pr["y"].is_number() && pr["z"].is_number())
                {
                    cmdOut.hasPos = true;
                    cmdOut.x = pr["x"].get<float>();
                    cmdOut.y = pr["y"].get<float>();
                    cmdOut.z = pr["z"].get<float>();
                }
                if (pr.contains("name") && pr["name"].is_string())
                    cmdOut.param = pr["name"].get<std::string>();
            }
        }
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
            std::string name = cmd.param.empty() ? "wave" : cmd.param;
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
