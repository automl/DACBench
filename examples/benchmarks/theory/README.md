This folder contains scripts for running DAC-DDQN experiments on RLS for LeadingOne.

#### Code: 
Besides the DACBench code, all neccessary source code are in `scripts`.

#### Data: 
DDQN results used in the paper are available in `experiments/results`.


#### Installation:
Please follow the installation guidelines of DACBench (see `README_DACBench.md` in the repo's home folder)

#### Run experiments:

- Steps to run a DDQN training:

```
export DAC=<DACBench_repo_folder>
source $DAC/start.sh
cd $DAC/rls_lo/rl/experiments/
mkdir ddqn
python ../scripts/run_experiment.py --out-dir ddqn/ --setting-file train_conf.yml
```

The command above will train a DDQN agent with settings specified in `experiments/train_conf.yml` (current setting: `n=50, k=3`, `evenly_spread`). To switch to another setting or to change the hyper-parameters of DDQN, please change the options inside `experiments/train_conf.yml` accordingly. For example, to train a DDQN agent with `n=150`, `k=3`, and with `initial_segment` portfolio setting, you can update the following fields:

```
bench:
    action_choices:         [1,2,3]
    instance_set_path:      "lo_rls_150"    
```

- Steps to evaluate a trained DDQN policy:

```
python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy DQNPolicy --policy-params results/n50_evenly_spread/k3/trained_ddqn/best.pt --out results/n50_evenly_spread/k3/dqn.pkl >results/n50_evenly_spread/k3/ddqn.txt
```
The command above will do 2000 repeated runs of RLS on LeadingOne using the trained DDQN agent located in `n50_evenly_spread/k3/ddqn/best.pt`

- Steps to evaluate random/optimal policies:
```
python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy RandomPolicy --out results/n50_evenly_spread/k3/random.pkl >results/n50_evenly_spread/k3/random.txt

python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy RLSOptimalDiscretePolicy --out results/n50_evenly_spread/k3/optimal.pkl >results/n50_evenly_spread/k3/optimal.txt
```

#### DDQN hyper-parameters:

We built our choice of DDQN hyperparameters based on prior literature using RL for dynamic tuning, in particular, each Q-networks has two hidden layers with 50 units (per layer) and ReLU activation function. The epsilon-greedy value is set to 0.2 (default value for DDQN in many cases). To allow the agent to collect enough information before the learning starts, 10000 random steps are done at the beginning of the training phase. 

The batch size and and the discount factor are set based on a small prestudy with the evenly spread portfolio setting where n=50, k=3. We tried two batch sizes: 2048 and 8192, both of which were able to reach the optimal policy several times during the training. Based on that result, we decided to choose batch size of 2048 for all experiments due to computational resource constraints. For the discount factor, we tried two values: 0.99 (default value for DDQN in many cases) and 0.9998. The former one resulted in much less stable learning compared to the latter one. This can be explained by the fact that the episode lengths of our benchmark are typically large (several thousands of steps) and most of the important progress is made during the later parts of an episode. Therefore, the discount factor of 0.9998 was finally chosen for all experiments in the paper.

