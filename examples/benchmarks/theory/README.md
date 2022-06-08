## Randomised Local Search (RLS) for the LeadingOnes problem as a Dynamic Algorithm Configuration (DAC) benchmark

This folder contains example scripts for running experiments on the DAC-LeadingOnes benchmark, and some DDQN results. For more information about the benchmark and the DDQN results, please see our paper ([arxiv](https://arxiv.org/abs/2202.03259), accepted at GECCO 2022) and the accompanying [blog post](https://andrebiedenkapp.github.io/blog/2022/gecco/) (written by André).

And [here](https://andrebiedenkapp.github.io/blog/2022/gecco/) is a blog post summarising the content of the paper.

If you use this benchmark, please cite us:
```
@article{biedenkapp2022theory,
  title={Theory-inspired Parameter Control Benchmarks for Dynamic Algorithm Configuration},
  author={Biedenkapp, Andr{\'e} and Dang, Nguyen and Krejca, Martin S and Hutter, Frank and Doerr, Carola},
  journal={arXiv preprint arXiv:2202.03259},
  year={2022},
  doi={https://doi.org/10.48550/arXiv.2202.03259}
}
```

### 1. Benchmark Description:

The RLS algorithm only has one parameter to be controlled, namely *r*. This is the number of bits being flipped at each iteration of RLS. The aim is to find a policy that decide which *r* to be used at each iteration such that the total number of evaluations required for reaching the optimal solution is minimised. 

The benchmark is configurable with multiple settings available:

**Setting 1: DAC-LeadingOnes with discrete action space**

This is the setting used in our paper mentioned above, where a list of possible values for *r* (a portfolio) is given as the action space. A portfolio *K* is defined via three properties:
- problem size *n*
- portfolio size *k*
- type of portfolio: how *k* values of *r* are chosen to added to the portfolio. There are three portfolio types: 
    + *evenly_spread*: *K* = {𝑖 · ⌊𝑛/𝑘⌋ + 1 | 𝑖 ∈ [0..𝑘 − 1]},
    + *initial_segment*: *K* = *[ 1..𝑘 ]*,
    + and *powers of 2*: *K* = {$2^𝑖$ | 𝑖 ≤ 𝑛 ∧ 𝑖 ∈ [0..𝑘 − 1]}

In our experiments, increasing either *n* or *k* would make it more difficult for the DDQN agent to learn a near-optimal policy. Even with the same *n* and *k*, switching between different portfolio types also has an effect on the learning behaviour. For more details, please see Section 4.4 in our paper. 

**Setting 2: DAC-LeadingOnes with continuous action space**

It is possible to use this benchmark with a continuous action space. In such setting, *r* is an integer value chosen from the range of *[1..n]*. 

**Initial solution**

For both settings, instead of starting each RLS run from a random initial solution, it is possible to start from a specific objective function value. This gives us another factor for controlling the difficult of the benchmark: starting from a very good solution would make the averge episode length shorter and make it easier for the RL agent to learn.


### 2. Getting Started:

A benchmark is defined via a configuration object (a dictionary). The following code create a benchmark with discrete action space (setting 1), with *n=50*, *k=5*, *powers of 2* portfolio type, and a cutoff time of 1e5 evaluations for each episode. Various initial objective functions are used and are specified in `<DACBench>/dacbench/instance_sets/theory/lo_rls_50.csv` (the code will automatically look into that folder if it cannot find the instance file in the current running folder).

```
from dacbench.benchmarks import TheoryBenchmark
bench_config = {"instance_set_path":"lo_rls_50.csv",
                "discrete_action": True,
                "cutoff": 1e5,
                "action_choices": [1,2,4,8,16]}
bench = TheoryBenchmark(config=bench_config)
env = bench.get_environment(test_env=True)                
```

Note that `test_env` indicates whether we are using this environment for training an RL agent or for evaluating a (learnt or baseline) policy. The difference between `test_env=False` and `test_env=True` is that with the former one (used for training): (i) cutoff time for an episode is set to 0.8*n^2 (n: problem size); and (ii) if an action is out of range, we stop the episode immediately and return a large negative reward (see `envs/theory.py` for more details), while the latter one means that the benchmark's original cutoff time is used, and out-of-range action will be clipped to nearest valid value and the episode will continue.

For a full example of how to run the benchmark with various settings, please have a look at the five examples in `examples.py`. To run those examples, set the `PYTHONPATH` environment variable to your DACBench folder:
```
export PYTHONPATH=<your_DACBench_folder>/:$PYTHONPATH
```

The five examples demonstrate how to:

- Create a benchmark with different settings as described above. 
- Evaluate a random policy and the optimal policies (discrete and non-discrete version) for a particular benchmark.
- Train a DDQN agent, evaluate the learnt agent and compare it with the optimal policy of the same setting.
- Calculate runtime (mean/std) of a policy without having to run the policy. The calculation is done using formulas derived from theoretical results.

### 3. Running multiple experiments:

We also provide scripts for reproducing all experiments used in our paper, and the results of our experiments. Please see the script 

- To run a DDQN training:

```
cd experiments/
mkdir ddqn
python ../scripts/run_experiment.py --out-dir ddqn/ --setting-file train_conf.yml
```

The command above will train a DDQN agent with settings specified in `experiments/train_conf.yml` (current setting: `n=50, k=3`, `evenly_spread`). To switch to another setting or to change the hyper-parameters of DDQN, please change the options inside `experiments/train_conf.yml` accordingly. For example, to train a DDQN agent with `n=150`, `k=3`, and with `initial_segment` portfolio setting, you can update the following fields:

```
bench:
    action_choices:         [1,2,3]
    instance_set_path:      "lo_rls_150.csv"    
```

- To evaluate a trained DDQN policy:

```
cd experiments/
python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy DQNPolicy --policy-params results/n50_evenly_spread/k3/trained_ddqn/best.pt --out results/n50_evenly_spread/k3/dqn.pkl
```
The command above will do 2000 repeated runs of RLS on LeadingOne using the trained DDQN agent located in `n50_evenly_spread/k3/ddqn/best.pt`

- Steps to evaluate random/optimal policies:
```
python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy RandomPolicy --out results/n50_evenly_spread/k3/random.pkl

python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy RLSOptimalDiscretePolicy --out results/n50_evenly_spread/k3/optimal.pkl
```

#### 4. DDQN hyper-parameters:

We built our choice of DDQN hyperparameters based on prior literature using RL for dynamic tuning, in particular, each Q-networks has two hidden layers with 50 units (per layer) and ReLU activation function. The epsilon-greedy value is set to 0.2 (default value for DDQN in many cases). To allow the agent to collect enough information before the learning starts, 10000 random steps are done at the beginning of the training phase. 

The batch size and and the discount factor are set based on a small prestudy with the evenly spread portfolio setting where n=50, k=3. We tried two batch sizes: 2048 and 8192, both of which were able to reach the optimal policy several times during the training. Based on that result, we decided to choose batch size of 2048 for all experiments due to computational resource constraints. For the discount factor, we tried two values: 0.99 (default value for DDQN in many cases) and 0.9998. The former one resulted in much less stable learning compared to the latter one. This can be explained by the fact that the episode lengths of our benchmark are typically large (several thousands of steps) and most of the important progress is made during the later parts of an episode. Therefore, the discount factor of 0.9998 was finally chosen for all experiments in the paper.

