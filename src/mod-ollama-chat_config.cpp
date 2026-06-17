#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_rag.h"
#include "Config.h"
#include "Log.h"
#include "mod-ollama-chat_api.h"
#include <fmt/core.h>
#include <sstream>
#include <fstream>


// --------------------------------------------
// Distance/Range Configuration
// --------------------------------------------
float      g_SayDistance       = 30.0f;
float      g_YellDistance      = 100.0f;
float      g_RandomChatterRealPlayerDistance = 40.0f;
float      g_EventChatterRealPlayerDistance = 40.0f;

// --------------------------------------------
// Bot/Player Chatter Probability & Limits
// --------------------------------------------
// Per-channel-type reply chances
uint32_t   g_PlayerReplyChance_Say     = 90;
uint32_t   g_BotReplyChance_Say        = 10;
uint32_t   g_PlayerReplyChance_Channel = 50;
uint32_t   g_BotReplyChance_Channel    = 5;
uint32_t   g_PlayerReplyChance_Party   = 90;
uint32_t   g_BotReplyChance_Party      = 10;
uint32_t   g_PlayerReplyChance_Guild   = 70;
uint32_t   g_BotReplyChance_Guild      = 5;

uint32_t   g_MaxBotsToPick     = 2;
uint32_t   g_RandomChatterBotCommentChance   = 5;
uint32_t   g_RandomChatterMaxBotsPerPlayer   = 2;
uint32_t   g_EventChatterBotCommentChance    = 15;
uint32_t   g_EventChatterBotSelfCommentChance = 5;
uint32_t   g_EventChatterMaxBotsPerPlayer    = 2;

// --------------------------------------------
// Ollama LLM API Configuration
// --------------------------------------------
std::string g_OllamaUrl        = "http://localhost:11434/api/generate";
std::string g_OllamaModel      = "llama3.2:1b";
uint32_t    g_OllamaNumPredict = 40;
float       g_OllamaTemperature = 0.8f;
float       g_OllamaTopP = 0.95f;
float       g_OllamaRepeatPenalty = 1.1f;
uint32_t    g_OllamaNumCtx = 0;
uint32_t    g_OllamaNumThreads = 0;
std::string g_OllamaStop = "";
std::string g_OllamaSystemPrompt = "";
std::string g_OllamaSeed = "";

// --------------------------------------------
// Concurrency/Queueing
// --------------------------------------------
uint32_t    g_MaxConcurrentQueries = 0;

// --------------------------------------------
// Feature Toggles & Core Settings
// --------------------------------------------
bool        g_Enable                          = true;
bool        g_DisableRepliesInCombat          = true;
bool        g_EnableRandomChatter             = true;
bool        g_EnableEventChatter              = true;
bool        g_EnableRPPersonalities           = false;
bool        g_EnableWhisperReplies            = false;
bool        g_DebugEnabled                    = false;
bool        g_DebugShowFullPrompt             = false;

// --------------------------------------------
// Think Mode Support
// --------------------------------------------
bool g_ThinkModeEnableForModule = false;

// --------------------------------------------
// Random Chatter Timing
// --------------------------------------------
uint32_t    g_MinRandomInterval               = 45;
uint32_t    g_MaxRandomInterval               = 180;

// --------------------------------------------
// Conversation History Settings
// --------------------------------------------
uint32_t    g_MaxConversationHistory          = 5;
uint32_t    g_ConversationHistorySaveInterval = 10;

// --------------------------------------------
// Prompt Templates
// --------------------------------------------
std::string g_RandomChatterPromptTemplate;
std::vector<std::string> g_RandomChatterPromptVariations;
std::vector<std::string> g_RandomChatterQuestionVariations;
std::string g_EventChatterPromptTemplate;
std::string g_ChatPromptTemplate;
std::string g_ChatExtraInfoTemplate;
std::vector<std::string> g_AllowedActions;

// --------------------------------------------
// Personality and Prompt Data
// --------------------------------------------
std::unordered_map<uint64_t, std::string> g_BotPersonalityList;
std::unordered_map<std::string, std::string> g_PersonalityPrompts;
std::vector<std::string> g_PersonalityKeys;
std::vector<std::string> g_PersonalityKeysRandomOnly;
std::string g_DefaultPersonalityPrompt;

// --------------------------------------------
// Chat History Templates and Toggles
// --------------------------------------------
bool        g_EnableChatHistory = true;
std::string g_ChatHistoryHeaderTemplate;
std::string g_ChatHistoryLineTemplate;
std::string g_ChatHistoryFooterTemplate;

// --------------------------------------------
// Chatbot Snapshot Template
// --------------------------------------------
bool        g_EnableChatBotSnapshotTemplate  = false;
std::string g_ChatBotSnapshotTemplate;

// --------------------------------------------
// Conversation History Store and Mutex
// --------------------------------------------
std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::deque<std::pair<std::string, std::string>>>> g_BotConversationHistory;
std::mutex g_ConversationHistoryMutex;
time_t g_LastHistorySaveTime = 0;

// --------------------------------------------
// Bot-Player Sentiment Tracking System
// --------------------------------------------
bool        g_EnableSentimentTracking = true;
float       g_SentimentDefaultValue = 0.5f;              // Default sentiment value (0.5 = neutral)
float       g_SentimentAdjustmentStrength = 0.1f;        // How much to adjust sentiment per message
uint32_t    g_SentimentSaveInterval = 10;                // How often to save sentiment to DB (minutes)
std::string g_SentimentAnalysisPrompt = "Analyze the sentiment of this message: \"{message}\". Respond only with: POSITIVE, NEGATIVE, or NEUTRAL.";
std::string g_SentimentPromptTemplate = "Your relationship sentiment with {player_name} is {sentiment_value} (0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone and response.";

// In-memory sentiment storage and mutex
std::unordered_map<uint64_t, std::unordered_map<uint64_t, float>> g_BotPlayerSentiments;
std::mutex g_SentimentMutex;
time_t g_LastSentimentSaveTime = 0;

// --------------------------------------------
// RAG (Retrieval-Augmented Generation) System
// --------------------------------------------
bool        g_EnableRAG = false;
std::string g_RAGDataPath = "rag/";
uint32_t    g_RAGMaxRetrievedItems = 3;
float       g_RAGSimilarityThreshold = 0.3f;
std::string g_RAGPromptTemplate;

class OllamaRAGSystem;
OllamaRAGSystem* g_RAGSystem = nullptr;

// --------------------------------------------
// Blacklist: Prefixes for Commands (not chat)
// --------------------------------------------
std::vector<std::string> g_BlacklistCommands = {
    ".playerbots",
    "playerbot",
};

// --------------------------------------------
// Environment/Contextual Random Chatter Templates
// --------------------------------------------
std::vector<std::string> g_EnvCommentCreature;
std::vector<std::string> g_EnvCommentGameObject;
std::vector<std::string> g_EnvCommentEquippedItem;
std::vector<std::string> g_EnvCommentBagItem;
std::vector<std::string> g_EnvCommentBagItemSell;
std::vector<std::string> g_EnvCommentSpell;
std::vector<std::string> g_EnvCommentQuestArea;
std::vector<std::string> g_EnvCommentVendor;
std::vector<std::string> g_EnvCommentQuestgiver;
std::vector<std::string> g_EnvCommentBagSlots;
std::vector<std::string> g_EnvCommentDungeon;
std::vector<std::string> g_EnvCommentUnfinishedQuest;

// --------------------------------------------
// Guild-Specific Random Chatter Templates
// --------------------------------------------
std::vector<std::string> g_GuildEnvCommentGuildMember;
std::vector<std::string> g_GuildEnvCommentGuildRank;
std::vector<std::string> g_GuildEnvCommentGuildBank;
std::vector<std::string> g_GuildEnvCommentGuildMOTD;
std::vector<std::string> g_GuildEnvCommentGuildInfo;
std::vector<std::string> g_GuildEnvCommentGuildOnlineMembers;
std::vector<std::string> g_GuildEnvCommentGuildRaid;
std::vector<std::string> g_GuildEnvCommentGuildEndgame;
std::vector<std::string> g_GuildEnvCommentGuildStrategy;
std::vector<std::string> g_GuildEnvCommentGuildGroup;
std::vector<std::string> g_GuildEnvCommentGuildPvP;
std::vector<std::string> g_GuildEnvCommentGuildCommunity;

// --------------------------------------------
// Guild-Specific Random Chatter Configuration
// --------------------------------------------
bool        g_EnableGuildEventChatter             = true;
bool        g_EnableGuildRandomAmbientChatter      = true;
uint32_t    g_GuildRandomChatterChance             = 10;
uint32_t    g_GuildChatterBotCommentChance          = 25;
uint32_t    g_GuildChatterMaxBotsPerEvent           = 2;

// --------------------------------------------
// Guild-Specific Event Chatter Templates
// --------------------------------------------
std::string g_GuildEventTypeLevelUp = "";
std::string g_GuildEventTypeDungeonComplete = "";
std::string g_GuildEventTypeEpicGear = "";
std::string g_GuildEventTypeRareGear = "";
std::string g_GuildEventTypeGuildJoin = "";
std::string g_GuildEventTypeGuildLeave = "";
std::string g_GuildEventTypeGuildPromotion = "";
std::string g_GuildEventTypeGuildDemotion = "";
std::string g_GuildEventTypeGuildLogin = "";
std::string g_GuildEventTypeGuildAchievement = "";

// --------------------------------------------
// Event Chatter Templates
// --------------------------------------------
std::string g_EventTypeDefeated;           // "defeated"
std::string g_EventTypeDefeatedPlayer;     // "defeated player"
std::string g_EventTypePetDefeated;        // "pet defeated"
std::string g_EventTypeGotItem;            // "got item"
std::string g_EventTypeDied;               // "died"
std::string g_EventTypeCompletedQuest;     // "completed quest"
std::string g_EventTypeLearnedSpell;       // "learned spell"
std::string g_EventTypeRequestedDuel;      // "requested to duel"
std::string g_EventTypeStartedDueling;     // "started dueling"
std::string g_EventTypeWonDuel;            // "won duel against"
std::string g_EventTypeLeveledUp;          // "leveled up"
std::string g_EventTypeAchievement;        // "earned achievement"
std::string g_EventTypeUsedObject;         // "used object"

// Chance variables for normal events
int g_EventTypeDefeated_Chance = 0;
int g_EventTypeDefeatedPlayer_Chance = 0;
int g_EventTypePetDefeated_Chance = 0;
int g_EventTypeGotItem_Chance = 0;
int g_EventTypeDied_Chance = 0;
int g_EventTypeCompletedQuest_Chance = 0;
int g_EventTypeLearnedSpell_Chance = 0;
int g_EventTypeRequestedDuel_Chance = 0;
int g_EventTypeStartedDueling_Chance = 0;
int g_EventTypeWonDuel_Chance = 0;
int g_EventTypeLeveledUp_Chance = 0;
int g_EventTypeAchievement_Chance = 0;
int g_EventTypeUsedObject_Chance = 0;

// Chance variables for guild events
int g_GuildEventTypeEpicGear_Chance = 0;
int g_GuildEventTypeRareGear_Chance = 0;
int g_GuildEventTypeGuildJoin_Chance = 0;
int g_GuildEventTypeGuildLogin_Chance = 0;
int g_GuildEventTypeGuildLeave_Chance = 0;
int g_GuildEventTypeGuildPromotion_Chance = 0;
int g_GuildEventTypeGuildDemotion_Chance = 0;
int g_GuildEventTypeGuildAchievement_Chance = 0;
int g_GuildEventTypeLevelUp_Chance = 0;
int g_GuildEventTypeDungeonComplete_Chance = 0;

// Event Cooldown
uint32_t g_EventCooldownTime = 10;

// --------------------------------------------
// Channel Disable Settings
// --------------------------------------------
bool g_DisableForCustomChannels = false;
bool g_DisableForSayYell = false;
bool g_DisableForGuild = false;
bool g_DisableForParty = false;

// --------------------------------------------
// Typing Simulation Settings
// --------------------------------------------
bool g_EnableTypingSimulation = false;
uint32_t g_TypingSimulationBaseDelay = 1000;     // 1000ms base delay
uint32_t g_TypingSimulationDelayPerChar = 250;   // 250ms per character (4 chars/sec)


static std::vector<std::string> SplitString(const std::string& str, char delim)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim))
    {
        // Trim whitespace from token
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
            tokens.push_back(token.substr(start, end - start + 1));
    }
    return tokens;
}

// Load Bot Personalities from Database
void LoadBotPersonalityList()
{    
    // Let's make sure our user has sourced the required sql file to add the new table
    QueryResult tableExists = CharacterDatabase.Query("SELECT * FROM information_schema.tables WHERE table_schema = 'acore_characters' AND table_name = 'mod_ollama_chat_personality' LIMIT 1");
    if (!tableExists)
    {
        LOG_ERROR("server.loading", "[Ollama Chat] Please source the required database table first");
        return;
    }

    QueryResult result = CharacterDatabase.Query("SELECT guid,personality FROM mod_ollama_chat_personality");

    if (!result)
    {
        return;
    }
    if (result->GetRowCount() == 0)
    {
        return;
    }    

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Fetching Bot Personality List into array");
    }

    do
    {
        uint64_t personalityBotGUID = result->Fetch()[0].Get<uint64_t>();
        std::string personalityKey = result->Fetch()[1].Get<std::string>();
        g_BotPersonalityList[personalityBotGUID] = personalityKey;
    } while (result->NextRow());
}

std::string GetMultiLineConfigValue(const std::string& configFilePath, const std::string& key)
{
    std::ifstream infile(configFilePath);
    if (!infile) return "";

    std::string line;
    std::string value;
    bool foundKey = false;
    while (std::getline(infile, line))
    {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
        if (trimmed.empty() || trimmed[0] == '#')
            continue;
        size_t pos = trimmed.find('=');
        if (!foundKey && pos != std::string::npos) {
            std::string possibleKey = trimmed.substr(0, pos);
            possibleKey.erase(possibleKey.find_last_not_of(" \t\r\n") + 1);
            if (possibleKey == key) {
                foundKey = true;
                std::string afterEq = trimmed.substr(pos + 1);
                afterEq.erase(0, afterEq.find_first_not_of(" \t\r\n"));
                value += afterEq;
                continue;
            }
        }
        else if (foundKey) {
            // New config key or section
            if (trimmed.find('=') != std::string::npos && trimmed.find('[') == std::string::npos)
                break;
            if (!value.empty()) value += "\n";
            value += trimmed;
        }
    }

    return value;
}

void LoadOllamaChatConfig()
{
    g_SayDistance                     = sConfigMgr->GetOption<float>("OllamaBotControl.SayDistance", 30.0f);
    g_YellDistance                    = sConfigMgr->GetOption<float>("OllamaBotControl.YellDistance", 100.0f);
    
    // Load per-channel-type reply chances
    g_PlayerReplyChance_Say           = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.PlayerReplyChance.Say", 90);
    g_BotReplyChance_Say              = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.BotReplyChance.Say", 10);
    g_PlayerReplyChance_Channel       = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.PlayerReplyChance.Channel", 50);
    g_BotReplyChance_Channel          = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.BotReplyChance.Channel", 5);
    g_PlayerReplyChance_Party         = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.PlayerReplyChance.Party", 90);
    g_BotReplyChance_Party            = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.BotReplyChance.Party", 10);
    g_PlayerReplyChance_Guild         = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.PlayerReplyChance.Guild", 70);
    g_BotReplyChance_Guild            = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.BotReplyChance.Guild", 5);
    
    g_MaxBotsToPick                   = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.MaxBotsToPick", 2);
    g_OllamaUrl                       = sConfigMgr->GetOption<std::string>("OllamaBotControl.Url", "http://localhost:11434/api/generate");
    g_OllamaModel                     = sConfigMgr->GetOption<std::string>("OllamaBotControl.Model", "llama3.2:1b");
    g_OllamaNumPredict                = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.NumPredict", 40);
    g_OllamaTemperature               = sConfigMgr->GetOption<float>("OllamaBotControl.Temperature", 0.8f);
    g_OllamaTopP                      = sConfigMgr->GetOption<float>("OllamaBotControl.TopP", 0.95f);
    g_OllamaRepeatPenalty             = sConfigMgr->GetOption<float>("OllamaBotControl.RepeatPenalty", 1.1f);
    g_OllamaNumCtx                    = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.NumCtx", 0);
    g_OllamaNumThreads                = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.NumThreads", 0);
    g_OllamaStop                      = sConfigMgr->GetOption<std::string>("OllamaBotControl.Stop", "");
    g_OllamaSystemPrompt              = sConfigMgr->GetOption<std::string>("OllamaBotControl.SystemPrompt", "");
    g_OllamaSeed                      = sConfigMgr->GetOption<std::string>("OllamaBotControl.Seed", "");

    g_MaxConcurrentQueries            = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.MaxConcurrentQueries", 0);

    g_Enable                          = sConfigMgr->GetOption<bool>("OllamaBotControl.Enable", true);
    g_DisableRepliesInCombat          = sConfigMgr->GetOption<bool>("OllamaBotControl.DisableRepliesInCombat", true);
    g_EnableRandomChatter             = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableRandomChatter", true);
    g_EnableEventChatter              = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableEventChatter", true);
    g_EnableWhisperReplies            = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableWhisperReplies", false);

    g_DebugEnabled                    = sConfigMgr->GetOption<bool>("OllamaBotControl.DebugEnabled", false);
    g_DebugShowFullPrompt             = sConfigMgr->GetOption<bool>("OllamaBotControl.DebugShowFullPrompt", false);

    g_MinRandomInterval               = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.MinRandomInterval", 45);
    g_MaxRandomInterval               = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.MaxRandomInterval", 180);
    g_RandomChatterRealPlayerDistance = sConfigMgr->GetOption<float>("OllamaBotControl.RandomChatterRealPlayerDistance", 40.0f);
    g_RandomChatterBotCommentChance   = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.RandomChatterBotCommentChance", 25);
    g_RandomChatterMaxBotsPerPlayer   = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.RandomChatterMaxBotsPerPlayer", 2);

    g_EnableGuildRandomAmbientChatter = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableGuildRandomAmbientChatter", true);
    g_GuildRandomChatterChance        = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.GuildRandomChatterChance", 10);

    g_EventChatterRealPlayerDistance = sConfigMgr->GetOption<float>("OllamaBotControl.EventChatterRealPlayerDistance", 40.0f);
    g_EventChatterBotCommentChance   = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.EventChatterBotCommentChance", 15);
    g_EventChatterBotSelfCommentChance = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.EventChatterBotSelfCommentChance", 5);
    g_EventChatterMaxBotsPerPlayer   = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.EventChatterMaxBotsPerPlayer", 2);

    g_EnableRPPersonalities           = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableRPPersonalities", false);

    g_RandomChatterPromptTemplate     = sConfigMgr->GetOption<std::string>("OllamaBotControl.RandomChatterPromptTemplate", "");

    // Load random chatter prompt variations
    std::string variationsStr = sConfigMgr->GetOption<std::string>("OllamaBotControl.RandomChatterPromptVariations", "");
    g_RandomChatterPromptVariations.clear();
    if (!variationsStr.empty())
    {
        std::stringstream ss(variationsStr);
        std::string variation;
        while (std::getline(ss, variation, '|'))
        {
            if (!variation.empty())
            {
                g_RandomChatterPromptVariations.push_back(variation);
            }
        }
    }

    // Load random chatter question variations
    std::string questionsStr = sConfigMgr->GetOption<std::string>("OllamaBotControl.RandomChatterQuestionVariations", "");
    g_RandomChatterQuestionVariations.clear();
    if (!questionsStr.empty())
    {
        std::stringstream ss(questionsStr);
        std::string question;
        while (std::getline(ss, question, '|'))
        {
            if (!question.empty())
            {
                g_RandomChatterQuestionVariations.push_back(question);
            }
        }
    }

    g_EventChatterPromptTemplate     = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventChatterPromptTemplate", "");

    g_ChatPromptTemplate              = sConfigMgr->GetOption<std::string>("OllamaBotControl.ChatPromptTemplate", "");
    
    g_ChatExtraInfoTemplate           = sConfigMgr->GetOption<std::string>("OllamaBotControl.ChatExtraInfoTemplate", "");

    g_DefaultPersonalityPrompt        = sConfigMgr->GetOption<std::string>("OllamaBotControl.DefaultPersonalityPrompt", "");

    g_MaxConversationHistory          = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.MaxConversationHistory", 5);
    g_ConversationHistorySaveInterval = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.ConversationHistorySaveInterval", 10);

    g_ChatHistoryHeaderTemplate       = sConfigMgr->GetOption<std::string>("OllamaBotControl.ChatHistoryHeaderTemplate", "");
    g_ChatHistoryLineTemplate         = sConfigMgr->GetOption<std::string>("OllamaBotControl.ChatHistoryLineTemplate", "");
    g_ChatHistoryFooterTemplate       = sConfigMgr->GetOption<std::string>("OllamaBotControl.ChatHistoryFooterTemplate", "");

    g_EnableChatBotSnapshotTemplate   = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableChatBotSnapshotTemplate", false);
    g_ChatBotSnapshotTemplate         = sConfigMgr->GetOption<std::string>("OllamaBotControl.ChatBotSnapshotTemplate", "");

    g_EnableChatHistory               = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableChatHistory", true);

    // Bot-Player Sentiment Tracking
    g_EnableSentimentTracking         = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableSentimentTracking", true);
    g_SentimentDefaultValue           = sConfigMgr->GetOption<float>("OllamaBotControl.SentimentDefaultValue", 0.5f);
    g_SentimentAdjustmentStrength     = sConfigMgr->GetOption<float>("OllamaBotControl.SentimentAdjustmentStrength", 0.1f);
    g_SentimentSaveInterval           = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.SentimentSaveInterval", 10);
    g_SentimentAnalysisPrompt         = sConfigMgr->GetOption<std::string>("OllamaBotControl.SentimentAnalysisPrompt", "Analyze the sentiment of this message: \"{message}\". Respond only with: POSITIVE, NEGATIVE, or NEUTRAL.");
    g_SentimentPromptTemplate         = sConfigMgr->GetOption<std::string>("OllamaBotControl.SentimentPromptTemplate", "Your relationship sentiment with {player_name} is {sentiment_value} (0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone and response.");

    // RAG (Retrieval-Augmented Generation) System
    g_EnableRAG                       = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableRAG", false);
    g_RAGDataPath                     = sConfigMgr->GetOption<std::string>("OllamaBotControl.RAGDataPath", "rag/");
    g_RAGMaxRetrievedItems            = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.RAGMaxRetrievedItems", 3);
    g_RAGSimilarityThreshold          = sConfigMgr->GetOption<float>("OllamaBotControl.RAGSimilarityThreshold", 0.3f);
    g_RAGPromptTemplate               = sConfigMgr->GetOption<std::string>("OllamaBotControl.RAGPromptTemplate", "RELEVANT INFORMATION:\n{rag_info}\nUse this information to provide accurate and detailed responses when applicable.");

    g_ThinkModeEnableForModule        = sConfigMgr->GetOption<bool>("OllamaBotControl.ThinkModeEnableForModule", false);

    // Typing Simulation
    g_EnableTypingSimulation          = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableTypingSimulation", false);
    g_TypingSimulationBaseDelay       = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.TypingSimulationBaseDelay", 1000);
    g_TypingSimulationDelayPerChar    = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.TypingSimulationDelayPerChar", 250);

    g_EventTypeDefeated           = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeDefeated", "");
    g_EventTypeDefeatedPlayer     = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeDefeatedPlayer", "");
    g_EventTypePetDefeated        = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypePetDefeated", "");
    g_EventTypeGotItem            = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeGotItem", "");
    g_EventTypeDied               = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeDied", "");
    g_EventTypeCompletedQuest     = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeCompletedQuest", "");
    g_EventTypeLearnedSpell       = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeLearnedSpell", "");
    g_EventTypeRequestedDuel      = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeRequestedDuel", "");
    g_EventTypeStartedDueling     = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeStartedDueling", "");
    g_EventTypeWonDuel            = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeWonDuel", "");
    g_EventTypeLeveledUp          = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeLeveledUp", "");
    g_EventTypeAchievement        = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeAchievement", "");
    g_EventTypeUsedObject         = sConfigMgr->GetOption<std::string>("OllamaBotControl.EventTypeUsedObject", "");


    // Load extra blacklist commands from config (comma-separated list)
    std::string extraBlacklist = sConfigMgr->GetOption<std::string>("OllamaBotControl.BlacklistCommands", "");
    if (!extraBlacklist.empty())
    {
        std::vector<std::string> extraList = SplitString(extraBlacklist, ',');
        for (const auto& cmd : extraList)
        {
            g_BlacklistCommands.push_back(cmd);
        }
    }

    // Allow-list of LLM action command types (comma-separated). Empty = allow the v1 set.
    {
        std::string allowed = sConfigMgr->GetOption<std::string>("OllamaBotControl.AllowedActions", "attack,follow,moveto,emote");
        g_AllowedActions = SplitString(allowed, ',');
    }

    LoadPersonalityTemplatesFromDB();

    g_queryManager.setMaxConcurrentQueries(g_MaxConcurrentQueries);

    // Loads the environment random chatter message templates for each type.
    // Each config option is a pipe-separated list of string templates,
    // using {} as a placeholder for named substitutions.
    // Helper to load a multi-line config option into a std::vector<std::string>
    auto LoadEnvCommentVector = [](const char* key, const std::vector<std::string>& defaults = {}) -> std::vector<std::string>
    {
        std::string val = sConfigMgr->GetOption<std::string>(key, "");
        std::vector<std::string> result;
        std::istringstream iss(val);
        std::string token;
        while (std::getline(iss, token, '|')) { // Split by '|'
            // Trim whitespace from token
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos)
                result.push_back(token.substr(start, end - start + 1));
        }
        if (result.empty() && !defaults.empty())
            return defaults;
        return result;
    };

    g_EnvCommentCreature        = LoadEnvCommentVector("OllamaBotControl.EnvCommentCreature", { "" });
    g_EnvCommentGameObject      = LoadEnvCommentVector("OllamaBotControl.EnvCommentGameObject", { "" });
    g_EnvCommentEquippedItem    = LoadEnvCommentVector("OllamaBotControl.EnvCommentEquippedItem", { "" });
    g_EnvCommentBagItem         = LoadEnvCommentVector("OllamaBotControl.EnvCommentBagItem", { "" });
    g_EnvCommentBagItemSell     = LoadEnvCommentVector("OllamaBotControl.EnvCommentBagItemSell", { "" });
    g_EnvCommentSpell           = LoadEnvCommentVector("OllamaBotControl.EnvCommentSpell", { "" });
    g_EnvCommentQuestArea       = LoadEnvCommentVector("OllamaBotControl.EnvCommentQuestArea", { "" });
    g_EnvCommentVendor          = LoadEnvCommentVector("OllamaBotControl.EnvCommentVendor", { "" });
    g_EnvCommentQuestgiver      = LoadEnvCommentVector("OllamaBotControl.EnvCommentQuestgiver", { "" });
    g_EnvCommentBagSlots        = LoadEnvCommentVector("OllamaBotControl.EnvCommentBagSlots", { "" });
    g_EnvCommentDungeon         = LoadEnvCommentVector("OllamaBotControl.EnvCommentDungeon", { "" });
    g_EnvCommentUnfinishedQuest = LoadEnvCommentVector("OllamaBotControl.EnvCommentUnfinishedQuest", { "" });

    // Guild-specific random chatter templates
    g_GuildEnvCommentGuildMember = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildMember", { "" });
    g_GuildEnvCommentGuildRank = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildRank", { "" });
    g_GuildEnvCommentGuildBank = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildBank", { "" });
    g_GuildEnvCommentGuildMOTD = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildMOTD", { "" });
    g_GuildEnvCommentGuildInfo = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildInfo", { "" });
    g_GuildEnvCommentGuildOnlineMembers = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildOnlineMembers", { "" });
    g_GuildEnvCommentGuildRaid = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildRaid", { "" });
    g_GuildEnvCommentGuildEndgame = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildEndgame", { "" });
    g_GuildEnvCommentGuildStrategy = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildStrategy", { "" });
    g_GuildEnvCommentGuildGroup = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildGroup", { "" });
    g_GuildEnvCommentGuildPvP = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildPvP", { "" });
    g_GuildEnvCommentGuildCommunity = LoadEnvCommentVector("OllamaBotControl.GuildEnvCommentGuildCommunity", { "" });

    // Guild-specific configuration
    g_EnableGuildEventChatter = sConfigMgr->GetOption<bool>("OllamaBotControl.EnableGuildEventChatter", true);
    g_GuildChatterBotCommentChance = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.GuildChatterBotCommentChance", 25);
    g_GuildChatterMaxBotsPerEvent = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.GuildChatterMaxBotsPerEvent", 2);

    // Guild-specific event templates
    g_GuildEventTypeLevelUp = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeLevelUp", "");
    g_GuildEventTypeDungeonComplete = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeDungeonComplete", "");
    g_GuildEventTypeEpicGear = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeEpicGear", "");
    g_GuildEventTypeRareGear = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeRareGear", "");
    g_GuildEventTypeGuildJoin = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeGuildJoin", "");
    g_GuildEventTypeGuildLogin = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeGuildLogin", "");
    g_GuildEventTypeGuildLeave = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeGuildLeave", "");
    g_GuildEventTypeGuildPromotion = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeGuildPromotion", "");
    g_GuildEventTypeGuildDemotion = sConfigMgr->GetOption<std::string>("OllamaBotControl.GuildEventTypeGuildDemotion", "");

    // Load chance variables for normal events
    g_EventTypeDefeated_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeDefeated_Chance", 0);
    g_EventTypeDefeatedPlayer_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeDefeatedPlayer_Chance", 0);
    g_EventTypePetDefeated_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypePetDefeated_Chance", 0);
    g_EventTypeGotItem_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeGotItem_Chance", 0);
    g_EventTypeDied_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeDied_Chance", 0);
    g_EventTypeCompletedQuest_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeCompletedQuest_Chance", 0);
    g_EventTypeLearnedSpell_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeLearnedSpell_Chance", 0);
    g_EventTypeRequestedDuel_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeRequestedDuel_Chance", 0);
    g_EventTypeStartedDueling_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeStartedDueling_Chance", 0);
    g_EventTypeWonDuel_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeWonDuel_Chance", 0);
    g_EventTypeLeveledUp_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeLeveledUp_Chance", 0);
    g_EventTypeAchievement_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeAchievement_Chance", 0);
    g_EventTypeUsedObject_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.EventTypeUsedObject_Chance", 0);

    // Load chance variables for guild events
    g_GuildEventTypeEpicGear_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeEpicGear_Chance", 0);
    g_GuildEventTypeRareGear_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeRareGear_Chance", 0);
    g_GuildEventTypeGuildJoin_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeGuildJoin_Chance", 0);
    g_GuildEventTypeGuildLogin_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeGuildLogin_Chance", 0);
    g_GuildEventTypeGuildLeave_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeGuildLeave_Chance", 0);
    g_GuildEventTypeGuildPromotion_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeGuildPromotion_Chance", 0);
    g_GuildEventTypeGuildDemotion_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeGuildDemotion_Chance", 0);
    g_GuildEventTypeGuildAchievement_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeGuildAchievement_Chance", 0);
    g_GuildEventTypeLevelUp_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeLevelUp_Chance", 0);
    g_GuildEventTypeDungeonComplete_Chance = sConfigMgr->GetOption<int>("OllamaBotControl.GuildEventTypeDungeonComplete_Chance", 0);


    // Cooldown time for events
    g_EventCooldownTime = sConfigMgr->GetOption<uint32_t>("OllamaBotControl.EventCooldownTime", 10);

    // Channel disable settings
    g_DisableForCustomChannels = sConfigMgr->GetOption<bool>("OllamaBotControl.DisableForCustomChannels", false);
    g_DisableForSayYell = sConfigMgr->GetOption<bool>("OllamaBotControl.DisableForSayYell", false);
    g_DisableForGuild = sConfigMgr->GetOption<bool>("OllamaBotControl.DisableForGuild", false);
    g_DisableForParty = sConfigMgr->GetOption<bool>("OllamaBotControl.DisableForParty", false);

    LOG_INFO("server.loading",
             "[Ollama Chat] Config loaded: Enabled = {}, SayDistance = {}, YellDistance = {}, "
             "Reply Chances - Say: P{}%/B{}%, Channel: P{}%/B{}%, Party: P{}%/B{}%, Guild: P{}%/B{}%, MaxBotsToPick = {}, "
             "Url = {}, Model = {}, MaxConcurrentQueries = {}, EnableRandomChatter = {}, MinRandInt = {}, MaxRandInt = {}, RandomChatterRealPlayerDistance = {}, "
             "RandomChatterBotCommentChance = {}. MaxConcurrentQueries = {}. Extra blacklist commands: {}",
             g_Enable, g_SayDistance, g_YellDistance,
             g_PlayerReplyChance_Say, g_BotReplyChance_Say,
             g_PlayerReplyChance_Channel, g_BotReplyChance_Channel,
             g_PlayerReplyChance_Party, g_BotReplyChance_Party,
             g_PlayerReplyChance_Guild, g_BotReplyChance_Guild,
             g_MaxBotsToPick,
             g_OllamaUrl, g_OllamaModel, g_MaxConcurrentQueries,
             g_EnableRandomChatter, g_MinRandomInterval, g_MaxRandomInterval, g_RandomChatterRealPlayerDistance,
             g_RandomChatterBotCommentChance, g_MaxConcurrentQueries, extraBlacklist);
}

void LoadPersonalityTemplatesFromDB()
{
    g_PersonalityPrompts.clear();
    g_PersonalityKeys.clear();
    g_PersonalityKeysRandomOnly.clear();

    QueryResult result = CharacterDatabase.Query("SELECT `key`, `prompt`, `manual_only` FROM `mod_ollama_chat_personality_templates`");
    if (!result)
    {
        LOG_ERROR("server.loading", "[Ollama Chat] No personality templates found in the database!");
        return;
    }

    do
    {
        std::string key = (*result)[0].Get<std::string>();
        std::string prompt = (*result)[1].Get<std::string>();
        bool manualOnly = (*result)[2].Get<bool>();
        
        g_PersonalityPrompts[key] = prompt;
        g_PersonalityKeys.push_back(key);
        
        // Only add to random pool if not manual_only
        if (!manualOnly)
        {
            g_PersonalityKeysRandomOnly.push_back(key);
        }
    } while (result->NextRow());

    LOG_INFO("server.loading", "[Ollama Chat] Cached {} personalities ({} available for random assignment).", 
             g_PersonalityKeys.size(), g_PersonalityKeysRandomOnly.size());
}

void LoadBotConversationHistoryFromDB()
{
    QueryResult result = CharacterDatabase.Query(
        "SELECT bot_guid, player_guid, player_message, bot_reply FROM mod_ollama_chat_history ORDER BY timestamp ASC"
    );
    if (!result)
        return;

    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    g_BotConversationHistory.clear();

    do {
        uint64_t botGuid = (*result)[0].Get<uint64_t>();
        uint64_t playerGuid = (*result)[1].Get<uint64_t>();
        std::string playerMsg = (*result)[2].Get<std::string>();
        std::string botReply = (*result)[3].Get<std::string>();

        auto& playerHistory = g_BotConversationHistory[botGuid][playerGuid];
        playerHistory.push_back({ playerMsg, botReply });
        while (playerHistory.size() > g_MaxConversationHistory)
        {
            playerHistory.pop_front();
        }

    } while (result->NextRow());

}


// Definition of the configuration WorldScript.
OllamaChatConfigWorldScript::OllamaChatConfigWorldScript() : WorldScript("OllamaChatConfigWorldScript") { }

void OllamaChatConfigWorldScript::OnStartup()
{
    LoadOllamaChatConfig();
    LoadBotPersonalityList();
    LoadBotConversationHistoryFromDB();
    InitializeSentimentTracking();

    // Initialize RAG system if enabled
    if (g_EnableRAG) {
        if (g_RAGSystem) {
            delete g_RAGSystem;
        }
        g_RAGSystem = new OllamaRAGSystem();
        if (!g_RAGSystem->Initialize()) {
            LOG_ERROR("server.loading", "[Ollama Chat] Failed to initialize RAG system");
            delete g_RAGSystem;
            g_RAGSystem = nullptr;
        } else {
            LOG_INFO("server.loading", "[Ollama Chat] RAG system initialized successfully");
        }
    }
}

void OllamaChatConfigWorldScript::OnShutdown()
{
    // Clean up RAG system
    if (g_RAGSystem) {
        delete g_RAGSystem;
        g_RAGSystem = nullptr;
        LOG_INFO("server.loading", "[Ollama Chat] RAG system cleaned up");
    }
}
