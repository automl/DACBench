#ifndef SEARCH_ENGINES_RL_EAGER_SEARCH_H
#define SEARCH_ENGINES_RL_EAGER_SEARCH_H

#include "eager_search.h"

#include "rl_client.h"
#include "../utils/timer.h"

class Evaluator;
class PruningMethod;

namespace options {
class Options;
}

namespace rl_eager_search {
class RLEagerSearch : public eager_search::EagerSearch {

    // RL relevant
    rl_client::RLClient rl_client;
    int rl_control_interval;
    int rl_steps_until_control;
    std::string rl_answer;
    utils::Timer engine_timer;
    std::map<int,std::map<std::string, double>> open_lists_stats;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit RLEagerSearch(const options::Options &opts);
    virtual ~RLEagerSearch() = default;
};
}

#endif