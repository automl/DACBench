#include "rl_open_list.h"

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

namespace rl_open_list {
template<class Entry>
class RLOpenList : public OpenList<Entry> {
    bool handle_empty_list;
    shared_ptr<utils::RandomNumberGenerator> rng;
    vector<unique_ptr<OpenList<Entry>>> open_lists;

protected:
    virtual void do_insertion(EvaluationContext &eval_context,
                              const Entry &entry) override;

public:
    explicit RLOpenList(const Options &opts);
    virtual ~RLOpenList() override = default;

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
RLOpenList<Entry>::RLOpenList(const Options &opts) 
    : handle_empty_list(opts.get<bool>("handle_empty_list")),
      rng(utils::parse_rng_from_options(opts))
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
void RLOpenList<Entry>::do_insertion(
    EvaluationContext &eval_context, const Entry &entry) {
    for (const auto &sublist : open_lists)
        sublist->insert(eval_context, entry);
}

template<class Entry>
Entry RLOpenList<Entry>::remove_min() {
    int best = 1;
    assert(best != -1);
    const auto &best_list = open_lists[best];
    assert(!best_list->empty());
    return best_list->remove_min();
}

template<class Entry>
Entry RLOpenList<Entry>::remove_min(int choice) {
    // Open list is empty => random choice
    if (open_lists[choice]->empty()) {
        if (!handle_empty_list) {
           std::cout << "No solution - FAILED" << std::endl;
           utils::exit_with(utils::ExitCode::SEARCH_UNSOLVED_INCOMPLETE);
           exit(12); 
        }
        std::vector<int> choices;
        for (size_t i = 0; i < open_lists.size(); ++i) {
             if (!open_lists[i]->empty()) {
                 choices.push_back(i);
             }
        }
        choice = choices[(*rng)(choices.size())]; 
    }

    const auto &best_list = open_lists[choice];
    assert(!best_list->empty());
    return best_list->remove_min();
}

template<class Entry>
bool RLOpenList<Entry>::empty() const {
    for (const auto &sublist : open_lists)
        if (!sublist->empty())
            return false;
    return true;
}

template<class Entry>
void RLOpenList<Entry>::clear() {
    for (const auto &sublist : open_lists)
        sublist->clear();
}

template<class Entry>
void RLOpenList<Entry>::get_path_dependent_evaluators(
    set<Evaluator *> &evals) {
    for (const auto &sublist : open_lists)
        sublist->get_path_dependent_evaluators(evals);
}

template<class Entry>
bool RLOpenList<Entry>::is_dead_end(
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
bool RLOpenList<Entry>::is_reliable_dead_end(
    EvaluationContext &eval_context) const {
    for (const auto &sublist : open_lists)
        if (sublist->is_reliable_dead_end(eval_context))
            return true;
    return false;
}

template<class Entry>
void RLOpenList<Entry>::get_open_lists_statistics(std::map<int, std::map<std::string, double>>& open_lists_stats) const {
    if (open_lists_stats.size() == 0) {
        for (size_t i = 0; i < open_lists.size(); ++i) {
            open_lists_stats[i] = std::map<std::string, double>();
        }
    }
    for (size_t i = 0; i < open_lists.size(); ++i) {
        open_lists.at(i)->get_open_list_statistics(open_lists_stats[i]);
    }
}

RLOpenListFactory::RLOpenListFactory(const Options &options)
    : options(options) {
}

unique_ptr<StateOpenList>
RLOpenListFactory::create_state_open_list() {
    return utils::make_unique_ptr<RLOpenList<StateOpenListEntry>>(options);
}

unique_ptr<EdgeOpenList>
RLOpenListFactory::create_edge_open_list() {
    return utils::make_unique_ptr<RLOpenList<EdgeOpenListEntry>>(options);
}

static shared_ptr<OpenListFactory> _parse(OptionParser &parser) {
    parser.document_synopsis("RL open list",
                             "rl open lists.");
    parser.add_list_option<shared_ptr<OpenListFactory>>(
        "sublists",
        "open lists between which this one alternates");
    
    parser.add_option<bool>("handle_empty_list", "handles empty list with random choice. necessary for unsafe heuristics", "true");

    utils::add_rng_options(parser);
    
    Options opts = parser.parse();
    opts.verify_list_non_empty<shared_ptr<OpenListFactory>>("sublists");
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<RLOpenListFactory>(opts);
}

static Plugin<OpenListFactory> _plugin("rl", _parse);
}
