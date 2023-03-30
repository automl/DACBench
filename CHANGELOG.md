# 0.2.0

### Interface Update
The main change in this version is going from OpenAI's gym to the newer gymnasium version. The outward change is slight, but this interface is now **incompatible with version 0.1.0**. 
To adapt to this version, you'll mainly have to replace instances of 'done' for termination with two variables: 'terminated' indication algorithm termination and 'truncated' indicating a timeout.
Combined they're equal to the old 'done'.
Additonally, the default version of the environments is now available in the gym registry.

### Multi-Agent / Round Robin Control Option
We added more options for controlling several hyperparameters at once. Using the PettingZoo API, users can now select which hyperparameters to control and use a typical Multi-Agent RL interface to do it.
This should provide more freedom in how to solve the problem of scaling up to multiple hyperparameters.

### Package Versions
We updated all our dependencies for this release. Please note that this will likely influence benchmark behaviour, so **do not** directly compare performance of version 0.2.0 with 0.1.0!

### Benchmark Changes
The OneLL benchmark is not the Theory benchmark with a similar goal and setup, but a different base problem. 
For versioning reasons, we removed ModEA, the same problem should be covered by ModCMA.
We also add a toy benchmark for higher dimensional spaces, the Geometric Benchmark.

### Switches Docs to GitHub.io
The documentation is now hosted on GitHub.io instead of Read the Docs for versioning reasons. The old version should still be accessible, however.


# 0.1.0
### Added Benchmarks
New benchmarks include the ModCMA IOHExperimenter version of ModEA, the OneLL EA benchmark and a toy version of controlling SGD.

### Singularity Containers
For added reproducibility, we provide Singularity recipes for each benchmark. This way they can be run in containers.

## ConfigSpace Integration
Search Spaces can now be defined via ConfigSpace and are then automatically converted to gym spaces. 
This should make it easier to recover whcih actions correspond to which hyperparameters.

# 0.0.1
Initial Version