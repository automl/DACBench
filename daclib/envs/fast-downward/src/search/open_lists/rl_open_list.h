#ifndef OPEN_LISTS_RL_OPEN_LIST_H
#define OPEN_LISTS_RL_OPEN_LIST_H

#include "../open_list_factory.h"
#include "../option_parser_util.h"


/*
  RL Open list based on alternation open list
*/

namespace rl_open_list {
class RLOpenListFactory : public OpenListFactory {
    Options options;
public:
    explicit RLOpenListFactory(const Options &options);
    virtual ~RLOpenListFactory() override = default;

    virtual std::unique_ptr<StateOpenList> create_state_open_list() override;
    virtual std::unique_ptr<EdgeOpenList> create_edge_open_list() override;
};
}

#endif
