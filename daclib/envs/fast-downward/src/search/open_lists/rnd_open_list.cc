#include "rnd_open_list.h"

#include "../open_list.h"
#include "../option_parser.h"
#include "../plugin.h"

#include "../utils/memory.h"
#include "../utils/system.h"
#include "../utils/rng.h"
#include "../utils/rng_options.h"

#include <cassert>
#include <memory>
#include <vector>

using namespace std;
using utils::ExitCode;

namespace rnd_open_list {
template<class Entry>
class RndOpenList : public OpenList<Entry> {
    shared_ptr<utils::RandomNumberGenerator> rng;
    vector<unique_ptr<OpenList<Entry>>> open_lists;

protected:
    virtual void do_insertion(EvaluationContext &eval_context,
                              const Entry &entry) override;

public:
    explicit RndOpenList(const Options &opts);
    virtual ~RndOpenList() override = default;

    virtual Entry remove_min() override;
    virtual Entry remove_min(int choice) override;
    virtual bool empty() const override;
    virtual void clear() override;
    virtual void get_path_dependent_evaluators(
        set<Evaluator *> &evals) override;
    virtual bool is_dead_end(
        EvaluationContext &eval_context) const override;
    virtual bool is_reliable_dead_end(
        EvaluationContext &eval_context) const override;

    virtual void get_open_lists_statistics(std::map<int, std::map<std::string, double>>& open_lists_stats) const override;
};


template<class Entry>
RndOpenList<Entry>::RndOpenList(const Options &opts) 
    : rng(utils::parse_rng_from_options(opts))
{
    vector<shared_ptr<OpenListFactory>> open_list_factories(
        opts.get_list<shared_ptr<OpenListFactory>>("sublists"));
    open_lists.reserve(open_list_factories.size());
    int i = 0;
    for (const auto &factory : open_list_factories) {
        open_lists.push_back(factory->create_open_list<Entry>());
        std::cout << "Open-List " << i++ << ": " << open_lists.back()->get_description() << std::endl;
    }

}

template<class Entry>
void RndOpenList<Entry>::do_insertion(
    EvaluationContext &eval_context, const Entry &entry) {
    for (const auto &sublist : open_lists)
        sublist->insert(eval_context, entry);
}

template<class Entry>
Entry RndOpenList<Entry>::remove_min() {
    std::vector<int> choices;
    for (size_t i = 0; i < open_lists.size(); ++i) {
        if (!open_lists[i]->empty()) {
            choices.push_back(i);
        }
    }
    int choice = choices[(*rng)(choices.size())];
    // std::cout << "Choice: " << choice << std::endl;
    return open_lists[choice]->remove_min();
}

template<class Entry>
Entry RndOpenList<Entry>::remove_min(int /*choice*/) {
    return this->remove_min();
}

template<class Entry>
bool RndOpenList<Entry>::empty() const {
    for (const auto &sublist : open_lists)
        if (!sublist->empty())
            return false;
    return true;
}

template<class Entry>
void RndOpenList<Entry>::clear() {
    for (const auto &sublist : open_lists)
        sublist->clear();
}

template<class Entry>
void RndOpenList<Entry>::get_path_dependent_evaluators(
    set<Evaluator *> &evals) {
    for (const auto &sublist : open_lists)
        sublist->get_path_dependent_evaluators(evals);
}

template<class Entry>
bool RndOpenList<Entry>::is_dead_end(
    EvaluationContext &eval_context) const {
    // If one sublist is sure we have a dead end, return true.
    if (is_reliable_dead_end(eval_context))
        return true;
    // Otherwise, return true if all sublists agree this is a dead-end.
    for (const auto &sublist : open_lists)
        if (!sublist->is_dead_end(eval_context))
            return false;
    return true;
}

template<class Entry>
bool RndOpenList<Entry>::is_reliable_dead_end(
    EvaluationContext &eval_context) const {
    for (const auto &sublist : open_lists)
        if (sublist->is_reliable_dead_end(eval_context))
            return true;
    return false;
}

template<class Entry>
void RndOpenList<Entry>::get_open_lists_statistics(std::map<int, std::map<std::string, double>>& open_lists_stats) const {
    if (open_lists_stats.size() == 0) {
        for (size_t i = 0; i < open_lists.size(); ++i) {
            open_lists_stats[i] = std::map<std::string, double>();
        }
    }
    for (size_t i = 0; i < open_lists.size(); ++i) {
        open_lists.at(i)->get_open_list_statistics(open_lists_stats[i]);
    }
}

RndOpenListFactory::RndOpenListFactory(const Options &options)
    : options(options) {
}

unique_ptr<StateOpenList>
RndOpenListFactory::create_state_open_list() {
    return utils::make_unique_ptr<RndOpenList<StateOpenListEntry>>(options);
}

unique_ptr<EdgeOpenList>
RndOpenListFactory::create_edge_open_list() {
    return utils::make_unique_ptr<RndOpenList<EdgeOpenListEntry>>(options);
}

static shared_ptr<OpenListFactory> _parse(OptionParser &parser) {
    parser.document_synopsis("Rnd open list",
                             "rnd open lists.");
    parser.add_list_option<shared_ptr<OpenListFactory>>(
        "sublists",
        "open lists between which this one alternates");
    
    utils::add_rng_options(parser);
    
    Options opts = parser.parse();
    opts.verify_list_non_empty<shared_ptr<OpenListFactory>>("sublists");
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<RndOpenListFactory>(opts);
}

static Plugin<OpenListFactory> _plugin("rnd", _parse);
}
