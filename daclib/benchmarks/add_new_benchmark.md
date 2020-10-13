# How to add a new benchmark to DAClib
## 1. Write a benchmark file
This is where you specify available options for your environment.
That includes options for the observation_space, reward range and action space.
At least one for each of these is mandatory, if you include multiple options please make sure that you also specify how to switch between them in the environment, e.g. by adding a variable for this.
Please also make sure to include a maximum number of steps per episode.
To use some specific wrappers, additional information is required. An example is the progress tracking wrapper, for which either an optimal policy for each instance or a way to compute it has to be specified.
The current benchmark classes should give you an idea how detailed these options should be.
We enourage you to provide as many possibilities to modify the benchmark as possible in order to make use of it in different scenarios.

## 2. Provide an environment & instance set (or a way to sample one)
Your environment should be a subclass of "AbstractEnv" which takes care of the instantiation of instance set and env descriptions.
To cycle through the instance set and reset the step counter, make sure to call super.reset_() each reset.
Checking for cutoff is done in super.step_() which will return if your episode is done yet. Feel free to specify other end conditions, this only looks at the number of steps.
Except for these methods, your env should simply follow the OpenAI gym standard.
Ideally, you would also use a relatively simple state representation to keep the environment user friendly (e.g. no dictionaries or nested lists).

As for instances, the standard way of using them in the benchmarks here is to read an instance set from file.
This is not mandatory, however! You can define an instance sampling method to work with our instance sampling wrapper to generate instances every episode or you can sample the instance set once before creating the environment.
How exactly you deal with instances should be specified in the benchmark class when creating the environment. An example for the instance sampling wrapper can be found in the SigmoidBenchmark class.

## 3. Add an example use case & test cases
To make the new benchmark accessible to everyone, please provide a small example of training an optimizer on it.
It can be short, but it should show an special cases (e.g. CMA uses a dictionary as state representation, therefore the example shows a way to flatten it into an array).
Additionally, please provide test cases for your benchmark class to ensure the environment is created properly and methods like reading the instance set work as they should.

## 4. Submit a pull request
Once everything is working, we would be grateful if you want to share the benchmark!
Submit a pull request on GitHub in which you briefly describe the benchmark and why it is interesting.
Feel free to include details on how it was modelled and please cite the source.
