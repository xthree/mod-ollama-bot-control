#include "mod-ollama-chat_worldstate.h"
#include "mod-ollama-chat_config.h"

#include "Playerbots.h"   // umbrella: AiObjectContext, PlayerbotAI, PlayerbotMgr
#include "Player.h"
#include "Unit.h"
#include "ObjectGuid.h"

#include <fmt/core.h>
#include <sstream>

std::string BuildBotWorldStateContext(Player* bot)
{
    if (!bot)
        return "";

    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (!botAI)
        return "";

    std::ostringstream oss;
    int listed = 0;

    GuidVector units = botAI->GetAiObjectContext()->GetValue<GuidVector>("nearest npcs")->Get();
    for (ObjectGuid const& guid : units)
    {
        Unit* unit = botAI->GetUnit(guid);
        if (!unit || unit->isDead() || !unit->IsInWorld())
            continue;

        const char* tag = unit->IsHostileTo(bot) ? "ENEMY" : "NPC";
        float dist = bot->GetDistance(unit);
        oss << fmt::format("[{}] {} (guid:{}) {:.0f}y\n", tag, unit->GetName(),
                           guid.GetRawValue(), dist);

        if (++listed >= 15)
            break;
    }

    if (listed == 0)
        oss << "(no notable units nearby)\n";

    return oss.str();
}
