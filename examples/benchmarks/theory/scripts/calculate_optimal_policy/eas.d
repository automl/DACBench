module eas;

public import std.algorithm.iteration;
public import std.algorithm.mutation;
public import std.algorithm.sorting;
public import std.bitmanip;
public import std.conv;
public import std.file;
public import std.parallelism;
public import std.path;
public import std.random;
public import std.range;
public import std.stdio;
public import std.string;

alias Individual = BitArray;
alias MutationPicker = int delegate(in int);

// *********************
// ** Portfolio Stuff **
// *********************

/**
 * Determines an optimal portfolio via brute force. Note that 1 is always included in the portfolio.
 * Saves the data collected in a file.
 */
int[] bruteForceOptimalPortfolio(in int n, in int portfolioSize)
in
{
    assert(portfolioSize >= 1 && portfolioSize <= n, "The portfolio size is incorrect.");
}
body
{
    import data_management;
    const path = DirectoryType.INFORMATION ~ dirSeparator ~ "all_optimal_policies" ~ dirSeparator;
    mkdirRecurse(path);
    const fileName = "n=" ~ n.text() ~ "_" ~ "k=" ~ portfolioSize.text();
    auto file = File(path ~ fileName, "w");
    file.writeln("run_time portfolio");

    int[] bestPortfolio = [];
    auto bestRunTime = double.infinity;

    // Call with a portfolio of [1].
    void stackedForLoops(in int layers, int[] portfolio)
    {
        auto newPortfolio = portfolio;
        if(layers > 1)
        {
            foreach(mutationRate; portfolio[$ - 1] + 1 .. n + 1)
            {
                    stackedForLoops(layers - 1, newPortfolio ~ [mutationRate]);
            }
        }
        else
        {
            const currentRunTime = determineOptimalRunTime(newPortfolio, n);
            file.writeln(currentRunTime, " ", newPortfolio);
            if(currentRunTime < bestRunTime)
            {
                bestRunTime = currentRunTime;
                bestPortfolio = newPortfolio;
            }
        }
    }

    stackedForLoops(portfolioSize, [1]);

    return bestPortfolio;
}

/**
 * Determines the optimal run time for a given portfolio in a setting of size n.
 */
double determineOptimalRunTime(in int[] portfolioIn, in int n)
{
    // Sort the portfolio descendingly;
    auto portfolio = portfolioIn.dup;
    portfolio.sort!("a > b")();
    auto breakingPoints = portfolio.determineOptimalBreakingPoints(n);

    // Concatenate the portfolio with the breaking points.
    auto zippedArrays = zip([-1] ~ portfolio, breakingPoints)
        .uniq!((a, b) => a[1] == b[1])()
        .array;

    // Calculate the optimal run time.
    double optimalRunTime = 0.0;
    foreach(i; 0 .. zippedArrays.length - 1) // @suppress(dscanner.suspicious.length_subtraction)
    {
        auto currentNumberOfBitsToFlip = zippedArrays[i + 1][0];
        auto startingIndexExclusive = zippedArrays[i][1];
        auto endingIndexInclusive = zippedArrays[i + 1][1];
        foreach(fitness; startingIndexExclusive + 1 .. endingIndexInclusive + 1)
        {
            auto probabilityOfImprovement = improvementProbability(n, currentNumberOfBitsToFlip, fitness);
            optimalRunTime += 1.0 / probabilityOfImprovement;
        }
    }
    return 0.5 * optimalRunTime;
}

/**
 * Given a portfolio of decreasing mutation strengths, this function determines the breaking points
 * that result in the lowest possible expected run time.
 */
int[] determineOptimalBreakingPoints(in int[] portfolio, in int n)
in
{
    assert(portfolio.isSorted!("a > b"), "The portfolio is not sorted.");
    assert(portfolio.length >= 1 && portfolio.length <= n, "The portfolio size is incorrect.");
}
body
{
    auto portfolioCardinality = cast(int) portfolio.length;

    // Note that the first and the last breaking points are set. There is one more breaking point that
    // the size of the portfolio.
    auto breakingPoints = new int[](portfolioCardinality + 1);
    breakingPoints[0] = -1;
    breakingPoints[portfolioCardinality] = n - 1;
    foreach(i; 1 .. portfolioCardinality)
    {
        breakingPoints[i] = determineBreakingPoint(portfolio[i - 1], portfolio[i], n);
    }
    return breakingPoints;
}

/**
 * Determine the breaking point of two values. This is the largest index [0 .. n - 1] such that the larger input has
 * an improvement quality of at least the smaller input.
 */
int determineBreakingPoint(in int larger, in int smaller, in int n)
{
    int breakingPoint = 0;
    foreach(i; 1 .. n)
    {
        if(improvementProbability(n, larger, i) < improvementProbability(n, smaller, i))
        {
            break;
        }
        breakingPoint = i;
    }
    return breakingPoint;
}

/**
 * The probability of an improvement on LO with n bits when exactly k bits are flipped and the current fitness is i.
 */
double improvementProbability(in int n, in int k, in int i)
{
    double improvementProbability = (k + 0.0) / n;
    foreach(j; 1 .. k)
    {
        improvementProbability *= (n - j - i + 0.0) / (n - j);
    }
    return improvementProbability;
}


// ********************
// ** Schedule Stuff **
// ********************

/** 
 * Returns a delegate that chosses one of the mutation strengths of the portfolio uniformly at random.
 */
MutationPicker getUniformPicker(in int[] portfolio)
{
    int chooseUniformMutationStrength(in int)
    {
        auto indexOfChosenPosition = uniform(0, portfolio.length);
        return portfolio[indexOfChosenPosition];
    }
    return &chooseUniformMutationStrength;
}

/**
 * Returns a delegate that chooses for a given fitness the best mutation strength out of the given portfolio.
 */
MutationPicker getOptimalPicker(in int[] portfolio, in int n)
{
    // Remove the first entry of the optimal breaking points, which is always -1.
    auto optimalBreakingPoints = determineOptimalBreakingPoints(portfolio, n)[1 .. $];
    int chooseOptimalMutationStrength(in int fitness)
    {
        foreach(index, breakingPoint; optimalBreakingPoints)
        {
            if(fitness <= breakingPoint)
            {
                return portfolio[index];
            }
        }
        // Since the last entry of the breaking points is always n - 1, the loop
        // cannot be exited without the return statement.
        assert(false);
    }
    return &chooseOptimalMutationStrength;
}


// ***************
// ** RLS Stuff **
// ***************

/**
 * A variant of RLS that chooses each iteration how many bits to flip based on the provided delegate.
 * The delegate receives a fitness value as input and returns a number of bits to flip as output.
 */
int rlsWithSchedule(in int n, MutationPicker chooseMutationRate)
{
    auto individual = generateUniformIndividual(n);
    auto numberOfIterations = 0;
    while(!isOptimal(individual))
    {
        auto fitness = individual.leadingOnesFitness();
        auto numberOfBitsToFlip = chooseMutationRate(fitness);
        auto offspring = individual.generateOffspring(numberOfBitsToFlip);
        if(offspring.leadingOnesFitness() >= fitness)
        {
            individual = offspring;
        }
        numberOfIterations++;
    }
    return numberOfIterations;
}

/**
 * Generates an offspring of an individual by flipping exactly the specified number of bits.
 */
Individual generateOffspring(in Individual parent, in int numberOfBitsToFlip)
{
    // Generate a copy of the parent and flip the respective bits.
    auto offspring = parent.dup;
    kFlipMutation(offspring, numberOfBitsToFlip);
    return offspring;
}

/**
 * Flips exactly k bits in the provided individual.
 */
void kFlipMutation(ref Individual individual, size_t k)
{
    const n = individual.length;
    assert(k <= n, "Too many bits need to be flipped.");

    /*
        This function is optimized for performance.
        We use rejection sampling and try to check quickyl for duplicates
        without allocating too much memory.
    */
    auto indicesAlreadyChosen = new size_t[k];
    bool alreadyChosen;
    size_t indexToCheck;
    foreach(index; iota(k))
    {
        auto flipPosition = uniform(0, individual.length);
        if(index > 0)
        {
            do
            {
                alreadyChosen = false;
                for(indexToCheck = 0; indexToCheck < index; indexToCheck++)
                {
                    if(flipPosition == indicesAlreadyChosen[indexToCheck])
                    {
                        alreadyChosen = true;
                        flipPosition = uniform(0, individual.length);
                        break;
                    }
                }
            }
            while(alreadyChosen);
        }
        indicesAlreadyChosen[index] = flipPosition;
        individual[flipPosition] = 1 - individual[flipPosition];
    }
}

/**
 * Generates an individual whose bits are chosen uniformly at random.
 */
Individual generateUniformIndividual(in int n)
{
    auto individual = Individual(new bool[n]);
    foreach(ref bit; individual)
    {
        bit = ((uniform01() < 0.5) ? 0 : 1);
    }
    return individual;
}

/**
 * Returns the number of leading 1s of the given individual.
 */
int leadingOnesFitness(in Individual individual)
{
    auto fitness = 0;
    foreach(bit; individual)
    {
        if(bit == 0)
        {
            break;
        }
        fitness++;
    }
    return fitness;
}

/**
 * Returns whether the given individual is optimal for LeadingOnes.
 */
bool isOptimal(in Individual individual)
{
    return (individual.leadingOnesFitness == individual.length);
}