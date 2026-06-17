#ifndef MOD_OLLAMA_CHAT_API_H
#define MOD_OLLAMA_CHAT_API_H

#include <string>
#include <future>
#include "mod-ollama-chat_querymanager.h"

std::string QueryOllamaAPI(const std::string& prompt);

// Action path: returns the model's raw output (full JSON) without the chat
// extract-between-quotes post-processing. If formatSchema is non-empty it is
// sent as Ollama's `format` (a JSON Schema) to constrain generation. numPredict
// (>=0) overrides the token cap so the structured JSON always completes.
std::string QueryOllamaRawAPI(const std::string& prompt, const std::string& formatSchema = "", int numPredict = -1);

// Checks if an API response is valid (not an error message)
bool IsValidAPIResponse(const std::string& response);

// Submits a query to the API.
std::future<std::string> SubmitQuery(const std::string& prompt);

// Declare the global QueryManager variable.
extern QueryManager g_queryManager;

#endif // MOD_OLLAMA_CHAT_API_H
