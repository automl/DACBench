# How to add a new benchmark to DAClib
## 1. Write a config file
This is where you specify available options for your environment.
That includes options for the observation_space, reward range and action space.
At least one for each of these is mandatory, if you include multiple options please make sure that you also specify how to switch between them in the environment, e.g. by adding a variable for this.
Please also make sure to include a maximum number of steps per episode.
To use some specific wrappers, additional information is required. An example is the progress tracking wrapper, for which either an optimal policy for each instance or a way to compute it has to be specified.

Possibility 1: .txt config
In this model, each line can contain an attribute and options, like:
reward_min: 0, -1
reward_max: 1, 1

Pro: easily editable
Cons: hard to include free changes of action and observation space, bloated

Possibility 2: .json config
Here we save dicts, so this should maybe be done from the benchmark class?
Pro: more flexible with complex arguments
Con: not well readable or editable

Possibility 3: make benchmark subclasses configs
Pros: complete flexibility, possibly better interaction
Cons: needs either a lot of interaction or a separate config file for involved setups, more complex system overall

## 2. Provide an environment & instance set
Your environment should be a subclass of "AbstractEnv" which takes care of the instantiation of instance set and env descriptions.
To cycle through the instance set, make sure to call super.reset() each reset.
Except for this, your env should simply follow the OpenAI gym standard.

## 3. Add your environment to the env factory
