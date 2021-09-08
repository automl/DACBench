"""
Python wrapper classes for HyFlex (should be a separate file, or even project)
"""

from enum import Enum

from typing import List

import numpy as np

import requests


class ProblemDomain:
    """
    This class implements a generic python wrapper for HyFlex problem domains.
    """

    HeuristicType = Enum('HeuristicType', 'CROSSOVER LOCAL_SEARCH MUTATION OTHER RUIN_RECREATE')

    def __init__(self, domain: str, seed: int, host: str = "http://127.0.0.1:8080"):
        """
        Creates a new problem domain and creates a new random number generator using the seed provided. If
        the seed takes the value -1, the seed is generated taking the current System time. The random number generator
        is used for all stochastic operations, so the problem will be initialised in the same way if the seed is the
        same. Sets the solution memory size to 2.

        :param domain: the unqualified class name of the HyFlex domain to be wrapped, e.g., SAT, BinPacking, etc.
        :param seed: a random seed
        """
        self.domain = domain
        self.seed = seed
        self.host = host
        self.session = requests.Session()
        self.token = self.session.put(self.host + "/instantiate/" + domain + "/" + str(seed)).text


    def getHeuristicCallRecord(self) -> List[int]:
        """
        Shows how many times each low level heuristic has been called.

        :return: A list which contains an integer value for each low level heuristic, representing the number of times
            that heuristic has been called by the HyperHeuristic object.
        """
        return self.session.get(self.host + "/heuristic/record/call/" + self.token).json()


    def getHeuristicCallTimeRecord(self) -> List[int]:
        """
        Shows the total time that each low level heuristic has been operating on the problem.

        :return: A list which contains an integer value representing the total number of milliseconds used by each low
            level heuristic.
        """
        return self.session.get(self.host + "/heuristic/record/callTime/" + self.token).json()


    def setDepthOfSearch(self, depthOfSearch: float) -> None:
        """
        Sets the parameter specifying the extent to which a local search heuristic will modify the solution.
        This parameter is related to the number of improving steps to be completed by the local search heuristics.

        :param depthOfSearch: must be in the range 0 to 1. The initial value of 0.1 represents the default operation of
            the low level heuristic.
        :return: None
        """
        self.session.post(self.host + "/search/depth/" + self.token + "/" + str(depthOfSearch))


    def setIntensityOfMutation(self, intensityOfMutation: float) -> None:
        """
        Sets the parameter specifying the extent to which a mutation or ruin-recreate low level heuristic will mutate
        the solution. For a mutation heuristic, this could mean the range of new values that a variable can take,
        in relation to its current value. It could mean how many variables are changed by one call to the heuristic.
        For a ruin-recreate heuristic, it could mean the percentage of the solution that is destroyed and rebuilt.
        For example, a value of 0.5 may indicate that half the solution will be rebuilt by a RUIN_RECREATE heuristic.
        The meaning of this variable is intentionally vaguely stated, as it depends on the heuristic in question,
        and the problem domain in question.

        :param intensityOfMutation: must be in the range 0 to 1. The initial value of 0.1 represents the default
            operation of the low level heuristic.
        :return: None
        """
        self.session.post(self.host + "/mutationIntensity/" + self.token + "/" + str(intensityOfMutation))


    def getDepthOfSearch(self) -> float:
        """
        Gets the current intensity of mutation parameter.

        :return: the current value of the intensity of mutation parameter.
        """
        return self.session.get(self.host + "/search/depth/" + self.token).text


    def getIntensityOfMutation(self) -> float:
        """
        Gets the current intensity of mutation parameter.

        :return: the current value of the intensity of mutation parameter.
        """
        return float(self.session.get(self.host + "/mutationIntensity/" + self.token).text)


    def getHeuristicsOfType(self, heuristicType: HeuristicType) -> List[int]:
        """
        Gets an array of heuristicIDs of the type specified by heuristicType.

        :param heuristicType: the heuristic type.
        :return: A list containing the indices of the heuristics of the type specified. If there are no heuristics of
            this type it returns None.
        """
        return list(self.session.get(self.host + "/heuristic/" + self.token + "/" + str(heuristicType.name)).json())


    def getHeuristicsThatUseIntensityOfMutation(self) -> List[int]:
        """
        Gets an array of heuristicIDs that use the intensityOfMutation parameter

        :param heuristicType: the heuristic type.
        :return: An array containing the indexes of the heuristics that use the intensityOfMutation parameter, or None
            if there are no heuristics of this type.
        """
        return self.session.get(self.host + "/heuristic/mutationIntensity/" + self.token).json()


    def getHeuristicsThatUseDepthOfSearch(self) -> List[int]:
        """
        Gets an array of heuristicIDs that use the depthOfSearch parameter

        :param heuristicType: the heuristic type.
        :return: An array containing the indexes of the heuristics that use the depthOfSearch parameter, or None if
            there are no heuristics of this type.
        """
        return self.session.get(self.host + "/heuristic/depth/" + self.token).json()


    def loadInstance(self, instanceID: int) -> None:
        """
        Loads the instance specified by instanceID.

        :param instanceID: Specifies the instance to load. The ID's start at zero.
        :return: None
        """
        self.session.post(self.host + "/instance/" + self.token + "/" + str(instanceID))


    def setMemorySize(self, size: int) -> None:
        """
        Sets the size of the array where the solutions are stored. The default size is 2.

        :param size: The new size of the solution array.
        :return: None
        """
        self.session.post(self.host + "/memorySize/" + self.token + "/" + str(size))


    def initialiseSolution(self, index: int) -> None:
        """
        Create an initial solution at a specified position in the memory array. The method of initialising the solution
        depends on the specific problem domain, but it is a random process, which will produce a different solution
        each time. The initialisation process may randomise all of the elements of the problem, or it may use a
        constructive heuristic with a randomised input.

        :param index: The index of the memory array at which the solution should be initialised.
        :return: None
        """
        self.session.put(self.host + "/solution/init/" + self.token + "/" + str(index))


    def getNumberOfHeuristics(self) -> int:
        """
        Gets the number of heuristics available in this problem domain

        :return: The number of heuristics available in this problem domain
        """
        return int(self.session.get(self.host + "/heuristic/num/" + self.token).text)


    def applyHeuristicUnary(self, heuristicID: int, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        """
        Applies the heuristic specified by heuristicID to the solution at position solutionSourceIndex and places the
        resulting solution at position solutionDestinationIndex in the solution array. If the heuristic is a
        CROSSOVER type then the solution at solutionSourceIndex is just copied to solutionDestinationIndex.

        :param heuristicID: The ID of the heuristic to apply (starts at zero)
        :param solutionSourceIndex: The index of the solution in the memory array to which to apply the heuristic
        :param solutionDestinationIndex: The index in the memory array at which to store the resulting solution
        :return: the objective function value of the solution created by applying the heuristic
        """
        return float(self.session.post(self.host + "/heuristic/apply/" + self.token + "/" + str(heuristicID) + "/" + str(
            solutionSourceIndex) + "/" + str(solutionDestinationIndex)).text)


    def applyHeuristicBinary(self, heuristicID: int, solutionSourceIndex1: int, solutionSourceIndex2: int,
                             solutionDestinationIndex: int) -> float:
        """
        Apply the heuristic specified by heuristicID to the solutions at position solutionSourceIndex1 and position
        solutionSourceIndex2 and put the resulting solution at position solutionDestinationIndex. The heuristic can
        be of any type (including CROSSOVER).

        :param heuristicID: The ID of the heuristic to apply (starts at zero)
        :param solutionSourceIndex1: The index of the first solution in the memory array to which to apply the heuristic
        :param solutionSourceIndex2: The index of the second solution in the memory array to which to apply the heuristic
        :param solutionDestinationIndex: The index in the memory array at which to store the resulting solution
        :return: the objective function value of the solution created by applying the heuristic
        """
        return float(self.session.post(self.host + "/heuristic/apply/" + self.token + "/" + str(heuristicID) + "/" + str(
            solutionSourceIndex1) + "/" + str(solutionSourceIndex2) + "/" + str(solutionDestinationIndex)).text)


    def copySolution(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> None:
        """
        Copies a solution from one position in the solution array to another

        :param solutionSourceIndex: The position of the solution to copy
        :param solutionDestinationIndex: The position in the array to copy the solution to.
        :return: None
        """
        self.session.post(self.host + "/solution/copy/" + self.token + "/" + str(solutionSourceIndex) + "/" + str(
            solutionDestinationIndex))


    def toString(self) -> str:
        """
        Gets the name of the problem domain. For example, "Bin Packing"

        :return: the name of the ProblemDomain
        """
        return self.session.get(self.host + "/toString/" + self.token).text


    def getNumberOfInstances(self) -> int:
        """
        Gets the number of instances available in this problem domain

        :return: the number of instances available
        """
        return int(self.session.get(self.host + "/instances/" + self.token).text)


    def bestSolutionToString(self) -> str:
        """
        Returns the objective function value of the best solution found so far by the HyperHeuristic.

        :return: The objective function value of the best solution.
        """
        return self.session.get(self.host + "/solution/best/toString/" + self.token).text


    def getBestSolutionValue(self) -> float:
        """
        Returns the objective function value of the best solution found so far by the HyperHeuristic.

        :return: The objective function value of the best solution.
        """
        return float(self.session.get(self.host + "/solution/best/value/" + self.token).text)


    def solutionToString(self, solutionIndex: int) -> str:
        """
        Gets a String representation of a given solution in memory

        :param solutionIndex: The index of the solution of which a String representation is required
        :return: A String representation of the solution at solutionIndex in the solution memory
        """
        return self.session.get(self.host + "/solution/toString/" + self.token + "/" + str(solutionIndex)).text


    def getFunctionValue(self, solutionIndex: int) -> float:
        """
        Gets the objective function value of the solution at index solutionIndex

        :param solutionIndex: The index of the solution from which the objective function is required
        :return: A double value of the solution's objective function value.
        """
        # raise NotImplementedError
        return float(self.session.get(self.host + "/solution/functionValue/" + self.token + "/" + str(solutionIndex)).text)


    # return self.mem[solutionIndex]

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        """
        Compares the two solutions on their structure (i.e. in the solution space, not in the objective/fitness
        function space).

        :param solutionIndex1: The index of the first solution in the comparison
        :param solutionIndex2: The index of the second solution in the comparison
        :return: true if the solutions are identical, false otherwise.
        """
        return bool(self.session.get(
            self.host + "/solution/compare/" + self.token + "/" + str(solutionIndex1) + "/" + str(solutionIndex2)).text)


"""
Gym Environment for HyFlex
"""
from dacbench import AbstractEnv

H_TYPE = ProblemDomain.HeuristicType


class HyFlexEnv(AbstractEnv):
    """
    Environment to control the step size of CMA-ES
    """

    def __init__(self, config):
        """
        Initialize CMA Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(HyFlexEnv, self).__init__(config)
        self.seed(config["seed"])

        # some useful constants
        # solution memory indices
        self.mem_size = 3
        self.s_best = 0  # mem pos for best
        self.s_inc = 1  # mem pos for incumbent
        self.s_prop = 2  # mem pos for proposal
        # actions
        self.reject = 0  # action corresponding to reject
        self.accept = 1  # action corresponding to accept

        self.seed = config["seed"]

        # The following variables are (re)set at reset
        self.problem = None  # HyFlex ProblemDomain object ~ current DAC instance
        self.unary_heuristics = None  # indices for unary heuristics
        self.binary_heuristics = None  # indices for binary heuristics
        self.f_best = None  # fitness of current best
        self.f_prop = None  # fitness of proposal
        self.f_inc = None  # fitness of incumbent

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : list
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        done = super(HyFlexEnv, self).step_()

        if action == self.accept:
            # accept previous proposal as new incumbent
            self.problem.copySolution(self.s_prop, self.s_inc)
            self.f_inc = self.f_prop

        # generate a new proposal
        self.f_prop = self._generate_proposal()

        # calculate reward (note: assumes f_best is not yet updated!)
        reward = self.get_reward(self)

        # update best
        self._update_best()

        return self.get_state(self), reward, done, {}

    def reset(self):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(HyFlexEnv, self).reset_()

        domain, instance_index = self.instance
        # create problem domain
        self.problem = ProblemDomain(domain, self.seed)
        # classify heuristics as unary/binary
        self.unary_heuristics = self.problem.getHeuristicsOfType(H_TYPE.LOCAL_SEARCH)
        self.unary_heuristics += self.problem.getHeuristicsOfType(H_TYPE.MUTATION)
        self.unary_heuristics += self.problem.getHeuristicsOfType(H_TYPE.RUIN_RECREATE)
        self.unary_heuristics += self.problem.getHeuristicsOfType(H_TYPE.OTHER)
        self.binary_heuristics = self.problem.getHeuristicsOfType(H_TYPE.CROSSOVER)
        # load instance
        self.problem.loadInstance(instance_index)
        # initialise solution memory
        self.problem.setMemorySize(self.mem_size)
        self.problem.initialiseSolution(self.s_inc)
        self.problem.copySolution(self.s_inc, self.s_best)
        # initialise fitness best/inc
        self.f_best = self.problem.getFunctionValue(self.s_best)
        self.f_inc = self.f_best
        # generate a proposal
        self.f_prop = self._generate_proposal()
        # update best
        self._update_best()
        return self.get_state(self)

    def _generate_proposal(self):
        # select uniformly at random between 0-ary (re-init), 1-ary, 2-ary heuristics
        nary = self.np_random.choice([0] + [1] * len(self.unary_heuristics) + [2] * len(self.binary_heuristics))
        if nary == 0:
            self.problem.initialiseSolution(self.s_prop)
            f_prop = self.problem.getFunctionValue(self.s_prop)
        elif nary == 1:
            h = self.np_random.choice(self.unary_heuristics)
            f_prop = self.problem.applyHeuristicUnary(h, self.s_inc, self.s_prop)
        else:
            h = self.np_random.choice(self.binary_heuristics)
            # note: the best solution found thus far is used as 2nd argument for crossover
            f_prop = self.problem.applyHeuristicBinary(h, self.s_inc, self.s_best, self.s_prop)
        return f_prop

    def _update_best(self):
        if self.f_prop < self.f_best:
            self.problem.copySolution(self.s_prop, self.s_best)
            self.f_best = self.f_prop

    def close(self):
        """
        No additional cleanup necessary

        Returns
        -------
        bool
            Cleanup flag
        """
        return True

    def render(self, mode: str = "human"):
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        if mode != "human":
            raise NotImplementedError

        print("incumbent: {} \t proposed: {} \t best: {}".format(self.f_inc, self.f_prop, self.f_best))

    def get_default_reward(self, _):
        """
        Compute reward

        Returns
        -------
        float
            Reward

        """
        return max(self.f_best - self.f_prop, 0)

    def get_default_state(self, _):
        """
        Gather state description

        Returns
        -------
        dict
            Environment state

        """
        return {"f_delta": self.f_inc - self.f_prop}
