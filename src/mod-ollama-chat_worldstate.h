#ifndef MOD_OLLAMA_CHAT_WORLDSTATE_H
#define MOD_OLLAMA_CHAT_WORLDSTATE_H

#include <string>

class Player;

// Build a GUID-grounded context block listing nearby units (with real GUIDs,
// hostility tags, and distance) for the given bot, so the LLM can reference a
// concrete target ("that boar") by its real guid instead of inventing one.
//
// MUST be called on the world thread (it reads grid state via the bot's AI
// context). Slice C3a extends this with a bot self-state block.
std::string BuildBotWorldStateContext(Player* bot);

#endif // MOD_OLLAMA_CHAT_WORLDSTATE_H
