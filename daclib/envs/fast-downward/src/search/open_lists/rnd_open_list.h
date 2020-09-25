#ifndef OPEN_LISTS_RND_OPEN_LIST_H
#define OPEN_LISTS_RND_OPEN_LIST_H

#include "../open_list_factory.h"
#include "../option_parser_util.h"


/*
  RND Open list based on alternation open list
*/

namespace rnd_open_list {
class RndOpenListFactory : public OpenListFactory {
    Options options;
public:
    explicit RndOpenListFactory(const Options &options);
    virtual ~RndOpenListFactory() override = default;

    virtual std::unique_ptr<StateOpenList> create_state_open_list() override;
    virtual std::unique_ptr<EdgeOpenList> create_edge_open_list() override;
};
}

#endif
