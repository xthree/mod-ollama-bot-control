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

    // Only hostile, attackable units — what "attack" needs. Listing all nearby NPCs
    // pulls in critters (rabbits, etc.) that the small model then fixates on.
    GuidVector units = botAI->GetAiObjectContext()->GetValue<GuidVector>("nearest hostile npcs")->Get();
    for (ObjectGuid const& guid : units)
    {
        Unit* unit = botAI->GetUnit(guid);
        if (!unit || unit->isDead() || !unit->IsInWorld())
            continue;
        if (!bot->IsValidAttackTarget(unit))
            continue;

        float dist = bot->GetDistance(unit);
        oss << fmt::format("[ENEMY] {} (guid:{}) {:.0f}y\n", unit->GetName(),
                           guid.GetRawValue(), dist);

        if (++listed >= 12)
            break;
    }

    if (listed == 0)
        oss << "(no enemies nearby)\n";

    return oss.str();
}
