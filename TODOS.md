### TODO list
If you are currently taking care of an issue, please mark it here and update the list after finishing.
If it's a big change (e.g. a new benchmark), consider using a new branch.
Non-release ready chenges are on dev. Merge to master whenever you feel it's warranted.

#### High level:
* Benchmarks to include:
  - DL LR benchmark (Freiburg, should be finished early December)
  - BBO competition pipeline (Freiburg/Difan, ask after results are public)
  - Modular CMA-ES (code not public)
  - any other interesting ones should be cleared with Marius/Theresa first
* Useful features:
  - State information on demand
  - Speedup techniques from AC, e.g. adaptive capping, racing, ...
* User experience:
  - Intro to RL Notebook using our envs
  - Make Basic Demo less basic and maybe into a Collab
  - Make visuals nicer
  - Have an "execute and forget" run_DACBench() function (including result plotting, maybe even rankings)
  - Maintain examples regularly and improve readability & diversity
  - Maybe use lazy importing for examples to prevent so many packages needing to be installed
* Backend performance:
  - Benchmark wrapper slowdowns
  - Fix the janky port assignment in FD
  - Enable parallel CMA
  - Singularity containers for each benchmark (ask Katha for advice on this)

#### Current low level issues:
* Check installation method: is this a good/the best way to do it?
* Think about OpenAI listing [Theresa]
* Is python 3.9 a good idea?
* Maybe use DynaQ for one of the examples to cover more bases
* For plotting: maybe use pandas for data collection and plotting?
* Add plotting in the performance wrapper
* Do GitHub actions make sense at the moment? If yes, implement them
* Get test coverage to 100%
