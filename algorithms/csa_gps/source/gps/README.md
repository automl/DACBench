We built upon the GPS version as given by Li and Malik [[1]](#1)
The original GPS code of Li and Malik can be found at [https://www.math.ias.edu/~ke.li/downloads/lto_code.tar.gz](https://www.math.ias.edu/~ke.li/downloads/lto_code.tar.gz)

In a nutshell, we modified the GPS code to be able to continuously sample from the starting teacher. To this end we introduce a sampling rate that determines how often we use new samples generated from the starting policy.

<a id="1">[1]</a> 
Li, K., Malik, J.: Learning to optimize. In: Proceedings of the International
Conference on Learning Representations (ICLR’17) (2017), published on-
line: [iclr.cc](iclr.cc)

### Contents
```bash
  |-gps
  |  |-agent # code for the LTO agent  
  |  |  |-lto # code for the CMAES world and agent
  |  |-sample # handling the trajectory samples
  |  |-utility # utilities, including the handling logging and output
  |  |-proto # the protocol buffers
  |  |-algorithm
  |  |  |-cost # code for computing the cost of trajectories
  |  |  |-traj_opt # code for trajectory optimization
  |  |  |-policy # policies that are used to obtain samples (CSA, Linear Gaussian and NN)
  |  |  |-policy_opt # code for policy optimization
  |  |  |-dynamics # code for handling the dynamics
```
This *gps* directory contains the code to run LTO-CMA. The above file tree contains a tree of the directories it consists of, and short descriptions of the code they contain. The code in the directory is under a *GNU GENERAL PUBLIC LICENSE v3*, except the specific files mentioned in the Modifications section, which are under an *APACHE v2* license.

#### Modifications to GPS code
In order to implement our approach, we have made modifications to the GPS code provided by Li and Malik. Below is the file tree depicting the list of files that have been either added or modified. Newly created files fall under Apache 2.0 whereas modified files keep their GPLv3 license
```bash
  |-gps
  |  |-agent
  |  |  |-lto
  |  |  |  |-agent_cmaes.py (under Apache 2.0 license)
  |  |  |  |-cmaes_world.py (under Apache 2.0 license)
  |  |-sample
  |  |  |-sample.py
  |  |-utility
  |  |  |-display.py
  |  |-proto
  |  |  |-gps.pb2.py
  |  |-algorithm
  |  |  |-policy
  |  |  |  |- lin_gauss_init.py
  |  |  |  |- lin_gauss_policy.py
  |  |  |  |- csa_policy.py (under Apache 2.0 license)
  |  |  |-policy_opt
  |  |  |  |-lto_model.py
```

