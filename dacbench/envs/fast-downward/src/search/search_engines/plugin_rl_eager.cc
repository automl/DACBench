#include "rl_eager_search.h"
#include "search_common.h"

#include "../option_parser.h"
#include "../plugin.h"

using namespace std;

namespace plugin_rl_eager {
static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
    parser.document_synopsis("RL Eager best-first search", "");

    parser.add_option<shared_ptr<OpenListFactory>>("open", "open list");
    parser.add_option<bool>("reopen_closed",
                            "reopen closed nodes", "false");
    parser.add_option<shared_ptr<Evaluator>>(
        "f_eval",
        "set evaluator for jump statistics. "
        "(Optional; if no evaluator is used, jump statistics will not be displayed.)",
        OptionParser::NONE);
    parser.add_list_option<shared_ptr<Evaluator>>(
        "preferred",
        "use preferred operators of these evaluators", "[]");
    parser.add_option<int>("rl_client_port", "rl client port", "54321");
    parser.add_option<int>("rl_control_interval", "rl control interval", "0");

    SearchEngine::add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    shared_ptr<rl_eager_search::RLEagerSearch> engine;
    if (!parser.dry_run()) {
        engine = make_shared<rl_eager_search::RLEagerSearch>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin("rl_eager", _parse);
}
