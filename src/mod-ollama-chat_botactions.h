#ifndef MOD_OLLAMA_CHAT_BOTACTIONS_H
#define MOD_OLLAMA_CHAT_BOTACTIONS_H

#include <cstdint>
#include <string>

class Player;

// A single LLM-decided command parsed from the model's JSON response.
struct BotActionCommand
{
    std::string type;            // "attack" | "follow" | "moveto" | "emote" | "none"
    uint64_t    targetGuid = 0;  // attack target (raw ObjectGuid value)
    bool        hasPos = false;  // moveto coordinates present
    float       x = 0.0f, y = 0.0f, z = 0.0f;
    std::string param;           // emote name, etc.
};

// --- Per-bot opt-in (world-thread only; in-memory for C3) ---
bool IsBotActionOptIn(Player* bot);
void SetBotActionOptIn(Player* bot, bool optedIn);

// Build the LLM action prompt (world thread; reads nearby world state).
std::string BuildBotActionPrompt(Player* bot, Player* sender, const std::string& message);

// Parse a model response into a spoken reply + a command. Tolerant of prose
// around the JSON: extracts the first balanced {...} object. Returns true if a
// JSON object was found and parsed.
bool ParseBotActionResponse(const std::string& response, std::string& sayOut, BotActionCommand& cmdOut);

// Enqueue a resolved action for world-thread execution (safe to call from a
// worker thread — only this and the drain touch the queue).
void EnqueueBotAction(uint64_t botGuid, uint64_t senderGuid, int sourceLocal,
                      const std::string& say, const BotActionCommand& cmd);

// Execute all queued actions on the world thread. Call from a WorldScript::OnUpdate.
void DrainBotActionQueue();

#endif // MOD_OLLAMA_CHAT_BOTACTIONS_H
