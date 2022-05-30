import numpy as np


def improvement_probability(r: int, i: int, n: int) -> float:
    """Returns the probability to improve the LeadingOnes value.

    Assumes that the bits to be flipped are chosen uniformly at random.

    Parameters:
    r (int): The number of bits flipped by the mutation operator.

    i (int): The fitness before flipping bits.

    n (int): The number of bits in the bit string.

    Watch out:
    It is assumed that r ≤ n and i ≤ n. Further, all arguments are supposed
    to be non-negative.
    """
    assert i <= n
    assert r <= n

    return (r / n) * np.prod([(n - i - j) / (n - j) for j in range(1, r)])


def expected_run_time(p, n: int) -> float:
    """Returns the expected run time of an algorithm using policy p when run on
    LeadingOnes.

    Parameters:
    p (array of size n): The policy that returns for a given fitness
    value how many bits to flip, p[i] is the number of bits flipped for fitness value i

    n (int): The length of the bit strings of the LeadingOnes function.

    Watch out:
    It is assumed that the domain of p is [0..n-1] and that p returns values in
    [0..n].
    """

    for i in range(0, n):
        v = p[i]
        assert (v >= 0) and (v <= n)
        if v > (n - i):
            return np.inf

    return 0.5 * np.sum([1 / improvement_probability(p[i], i, n) for i in range(0, n)])


def variance_in_run_time(p, n: int) -> float:
    """Returns the variance of the run time of an algorithm using policy p
    when run on LeadingOnes.

    Parameters:
    p (array of size n): The policy that returns for a given fitness
    value how many bits to flip, p[i] is the number of bits flipped for fitness value i

    n (int): The length of the bit strings of the LeadingOnes function.

    Watch out:
    It is assumed that the domain of p is [0..n-1] and that p returns values in
    [0..n].
    """

    def variance_for_fitness(i: int) -> float:
        """A helper function that calculates the variance if the fitness is
        i."""
        q = improvement_probability(p[i], i, n)
        return (3 - 2 * q) / (q ** 2)

    for i in range(0, n):
        v = p[i]
        assert (v >= 0) and (v <= n)
        if v > (n - i):
            return np.inf

    return 0.25 * np.sum([variance_for_fitness(i) for i in range(0, n)])
