## Containerized Benchmarks

DACBench can run containerized versions of Benchmarks using Singularity containers to isolate their dependencies and make reproducible Singularity images. 

### Building a Container

For writing your own recipe to build a Container, you can refer to `dacbench/container/singularity_recipes/recipe_template`  

Install Singularity and run the following to build the (in this case) cma container

```bash
cd dacbench/container/singularity_recipes
sudo singularity build cma cma.def
```
