# How to add a new benchmark to DAClib
## 1. Write a benchmark file
This is where you specify available options for your environment.
That includes options for the observation_space, reward range and action space.
At least one for each of these is mandatory, if you include multiple options please make sure that you also specify how to switch between them in the environment, e.g. by adding a variable for this.
Please also make sure to include a maximum number of steps per episode.
To use some specific wrappers, additional information is required. An example is the progress tracking wrapper, for which either an optimal policy for each instance or a way to compute it has to be specified.
The current benchmark classes should give you an idea how detailed these options should be.
We enourage you to provide as many possibilities to modify the benchmark as possible in order to make use of it in different scenarios.

## 2. Provide an environment & instance set
Your environment should be a subclass of "AbstractEnv" which takes care of the instantiation of instance set and env descriptions.
To cycle through the instance set, make sure to call super.reset() each reset.
Except for this, your env should simply follow the OpenAI gym standard.

## 3. Add your environment to the env factory
