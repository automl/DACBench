## Containerized Benchmarks

DACBench can run containerized versions of Benchmarks using Singularity containers to isolate their dependencies and make reproducible Singularity images. 


### Building a Container

Install Singularity and run the following to build the cma container

```bash
cd dacbench/container/singularity_recipes
sudo singularity build cma cma.def
```
