module data_management;

public import eas;

/**
 * The different options for functions that return policies.
 */
enum PickerName
{
    RANDOM = "random",
    OPTIMAL = "optimal",
}

/**
 * The different options for policies to choose from.
 */
enum PortfolioName
{
    POWERS_OF_TWO = "powers_of_2",
    INITIAL_SEGMENT = "initial_segment",
    EVENLY_SPREAD = "evenly_spread",
    OPTIMAL = "optimal_portfolio",
}

/**
 * The different options for where data can lie. Each option denotes a top-level
 * directory that is then followed by the ›path‹ provided in ›FileData‹.
 */
enum DirectoryType
{
    EXPERIMENTS = "experiments",
    EVALUATION = "evaluation",
    INFORMATION = "information",
    VISUALIZATION = "visualization",
}

/**
 * The different options for what compound data to use.
 */
enum CompoundType
{
    POLICIES = "policies",
}

/**
 * The different options for what data is used for the visualization.
 */
enum VisualizationDataSource
{
    EMPIRICAL = "empirical",
    OPTIMAL = "optimal",
}

/**
 * The different options for what visualization is chosen.
 */
enum VisualizationType
{
    VARY_K = "vary_k",
    VARY_N = "vary_n",
}


// ******************
// ** Helper Stuff **
// ******************

/**
 * Used for communicating different cases to run or evaluate.
 */
struct TestCase
{
    int n;
    int portfolioSize;
    PortfolioName portfolioName;
    PickerName pickerName;
}

/**
 * Used for communicating different cases to visualize.
 */
struct PlotCase
{
    int[] nValues;
    int[] kValues;
    PortfolioName portfolioName;
    PickerName pickerName;
    VisualizationType visualizationType;
    VisualizationDataSource visualizationDataSource;
}

/**
 * Used for storing information about file locations.
 */
struct FileData
{
    string path;
    string fileName;
    string filePath;
}

/**
 * Returns file data for a specific setting, not requiring a PickerName.
 * Creates the directory if it is not already present.
 */
FileData getShortFileData(in DirectoryType directoryType, in int n, in int portfolioSize,
    in PortfolioName portfolioName)
{
    auto path = directoryType ~ dirSeparator ~ "k=" ~ portfolioSize.text() ~ dirSeparator;
    mkdirRecurse(path);
    auto fileName = "n=" ~ n.text() ~ "_" ~ portfolioName;
    auto filePath = path ~ fileName;
    return FileData(path, fileName, filePath);
}

/**
 * Returns file data for a specific setting, requiring a PickerName.
 * Creates the directory if it is not already present.
 */
FileData getFileData(in DirectoryType directoryType, in int n, in int portfolioSize, in PortfolioName portfolioName,
    in PickerName pickerName)
{
    auto fileData = getShortFileData(directoryType, n, portfolioSize, portfolioName);
    auto path = fileData.path;
    auto fileName = fileData.fileName ~ "_" ~ pickerName;
    auto filePath = path ~ fileName;
    return FileData(path, fileName, filePath);
}

/**
 * Returns file data for compound information.
 * Creates the directory if it is not already present.
 */
FileData getCompoundInformationFileData(in int n, in PortfolioName portfolioName, in CompoundType compoundType)
{
    auto path = DirectoryType.INFORMATION ~ dirSeparator ~ compoundType ~ dirSeparator;
    mkdirRecurse(path);
    auto fileName = "n=" ~ n.text() ~ "_" ~ portfolioName;
    auto filePath = path ~ fileName;
    return FileData(path, fileName, filePath);
}

/**
 * Returns file data for visualization purposes. Requires to determine the type of visualization and the data source.
 * Creates the directory if it is not already present.
 */
FileData getVisualizationFileData(in int fixedValue, in PortfolioName portfolioName, in PickerName pickerName,
    in VisualizationType visualizationType, in VisualizationDataSource visualizationDataSource)
{
    auto path = DirectoryType.VISUALIZATION ~ dirSeparator ~ "plot_data" ~ dirSeparator;
    auto fileName = "";
    final switch(visualizationType)
    {
        with(VisualizationType)
        {
            case VARY_K:
            case VARY_N:
                path ~= visualizationType ~ dirSeparator ~ visualizationDataSource ~ dirSeparator;
                fileName ~= ((visualizationType == VARY_K) ? "n" : "k");
                fileName ~= "=" ~ fixedValue.text() ~ "_" ~ portfolioName;
                if(visualizationDataSource == VisualizationDataSource.EMPIRICAL)
                {
                    fileName ~= "_" ~ pickerName;
                }
                break;
        }
    }
    mkdirRecurse(path);
    auto filePath = path ~ fileName;
    return FileData(path, fileName, filePath);
}

/**
 * Determines the portfolio, given the portfolio name. Returns it in descending order.
 */
int[] calculatePortfolio(in int n, in int portfolioSize, in PortfolioName portfolioName)
{
    int[] portfolio = ()
        {
            with(PortfolioName)
            {
                final switch(portfolioName)
                {
                    case POWERS_OF_TWO:
                        return iota(0, portfolioSize).map!(e => 2 ^^ e)().array();
                    case INITIAL_SEGMENT:
                        return iota(1, portfolioSize + 1).array();
                    case EVENLY_SPREAD:
                        return iota(0, portfolioSize).map!(e => e * (n / portfolioSize) + 1)().array();
                    case OPTIMAL:
                        return bruteForceOptimalPortfolio(n, portfolioSize);
                }
            }
        }();
    portfolio.sort!("a > b")();
    return portfolio;
}


// *******************
// ** Visualization **
// *******************

/** 
 * Produces visualization data for multiple tests at once, given as an array of test cases.
 * Requires results from runs to be present. If not, no evaluation is generated.
 */
void generateMultiplePlotData(in PlotCase[] plotCases)
{
    foreach(plotCase; plotCases)
    {
        generatePlotData(plotCase);
    }
}

/** 
 * Generates visualization data used for a plot. If one value is varied, for the other value, the first element
 * of the array is chosen.
 */
void generatePlotData(in PlotCase plotCase)
{
    final switch(plotCase.visualizationType)
    {
        with(VisualizationType)
        {
            case VARY_K:
                const n = plotCase.nValues[0];
                const visualizationFileData = getVisualizationFileData(n, plotCase.portfolioName,
                    plotCase.pickerName, plotCase.visualizationType, plotCase.visualizationDataSource);
                auto visualizationFile = File(visualizationFileData.filePath, "w");
                final switch(plotCase.visualizationDataSource)
                {
                    case VisualizationDataSource.EMPIRICAL:
                        visualizationFile.writeln("k avg lq med uq");
                        foreach(k; plotCase.kValues)
                        {
                            const evaluationFileData = getFileData(DirectoryType.EVALUATION, n, k,
                                plotCase.portfolioName, plotCase.pickerName);
                            if(!exists(evaluationFileData.filePath))
                            {
                                continue;
                            }
                            const evaluationContent = readText(evaluationFileData.filePath).split("\n")[1];
                            visualizationFile.write(k, " ", evaluationContent);
                        }
                        return;
                    case VisualizationDataSource.OPTIMAL:
                        visualizationFile.writeln("k rel-rt");
                        foreach(k; plotCase.kValues)
                        {
                            const informationFileData = getShortFileData(DirectoryType.INFORMATION, n, k,
                                plotCase.portfolioName);
                            if(!exists(informationFileData.filePath))
                            {
                                continue;
                            }
                            const optimalRunTime = readText(informationFileData.filePath).split("\n")[3]
                                .split(" ")[$ - 1].strip().to!(double);
                            visualizationFile.writeln(k, " ", optimalRunTime / (n ^^ 2));
                        }
                        return;
                }
                assert(false);
            case VARY_N:
                const portfolioSize = plotCase.kValues[0];
                const visualizationFileData = getVisualizationFileData(portfolioSize, plotCase.portfolioName,
                    plotCase.pickerName, plotCase.visualizationType, plotCase.visualizationDataSource);
                auto visualizationFile = File(visualizationFileData.filePath, "w");
                final switch(plotCase.visualizationDataSource)
                {
                    case VisualizationDataSource.EMPIRICAL:
                        visualizationFile.writeln("n avg lq med uq");
                        foreach(n; plotCase.nValues)
                        {
                            const evaluationFileData = getFileData(DirectoryType.EVALUATION, n, portfolioSize,
                                plotCase.portfolioName, plotCase.pickerName);
                            if(!exists(evaluationFileData.filePath))
                            {
                                continue;
                            }
                            const evaluationContent = readText(evaluationFileData.filePath).split("\n")[1];
                            visualizationFile.write(n, " ", evaluationContent);
                        }
                        return;
                    case VisualizationDataSource.OPTIMAL:
                        visualizationFile.writeln("n rel-rt");
                        foreach(n; plotCase.nValues)
                        {
                            const informationFileData = getShortFileData(DirectoryType.INFORMATION, n, portfolioSize,
                                plotCase.portfolioName);
                            if(!exists(informationFileData.filePath))
                            {
                                continue;
                            }
                            const optimalRunTime = readText(informationFileData.filePath).split("\n")[3]
                                .split(" ")[$ - 1].strip().to!(double);
                            visualizationFile.writeln(n, " ", optimalRunTime / (n ^^ 2));
                        }
                        return;
                }
                assert(false);
        }
    }
    assert(false);
}


// *****************
// ** Information **
// *****************

/**
 * Generates information for all the provided test cases.
 */
void generateInformation(in TestCase[] testCases)
{
    foreach(testCase; testCases)
    {
        calculateInformation(testCase.n, testCase.portfolioSize, testCase.portfolioName);
    }
}

/**
 * Generates compound information for all the provided values of n. For each, it does so for
 * all values of k.
 */
void generateCompoundInformation(in int[] nValues, in int[] kValues, in PortfolioName[] portfolioNames,
    in CompoundType compoundType)
{
    foreach(n; nValues)
    {
        foreach(portfolioName; portfolioNames)
        {
            gatherCompoundInformation(n, kValues, portfolioName, compoundType);
        }
    }
}

/**
 * Calculates for the specified setting what its portfolio, optimal policy, and optimal expected run time are. Writes them into a file.
 */
void calculateInformation(in int n, in int portfolioSize, in PortfolioName portfolioName)
{
    auto portfolio = calculatePortfolio(n, portfolioSize, portfolioName);
    // Abort if the portfolio is invalid.
    if(portfolio[0] > n)
    {
        return;
    }
    auto informationFileData = getShortFileData(DirectoryType.INFORMATION, n, portfolioSize, portfolioName);
    auto informationFile = File(informationFileData.filePath, "w");
    auto optimalBreakingPoints = determineOptimalBreakingPoints(portfolio, n)[1 .. $];
    auto optimalRunTime = determineOptimalRunTime(portfolio, n);
    informationFile.writeln("portfolio: ", portfolio);
    informationFile.writeln("optimal policy: ", optimalBreakingPoints);
    informationFile.writeln("optimal policy relative: ", optimalBreakingPoints.map!(point => point / (n + 0.0))());
    informationFile.writeln("optimal expected run time: ", optimalRunTime);
    informationFile.writeln("relative optimal expected run time: ", optimalRunTime / (n ^^ 2));
    informationFile.writef("%(%d,%); ", portfolio.dup.reverse());
    informationFile.writefln("%(%d,%)", optimalBreakingPoints.reverse());
}

/**
 * Gather for the specified compound type the respective information.
 */
void gatherCompoundInformation(in int n, in int[] kValues, in PortfolioName portfolioName, in CompoundType compoundType)
{
    auto compoundInformationFileData = getCompoundInformationFileData(n, portfolioName, compoundType);
    auto compoundInformationFile = File(compoundInformationFileData.filePath, "w");
    final switch(compoundType)
    {
        with(CompoundType)
        {
            case POLICIES:
                foreach(portfolioSize; kValues)
                {
                    auto informationFileData = getShortFileData(DirectoryType.INFORMATION, n, portfolioSize,
                        portfolioName);
                    if(!exists(informationFileData.filePath))
                    {
                        continue;
                    }
                    auto policyData = readText(informationFileData.filePath).split("\n")[5];
                    compoundInformationFile.write(policyData);
                }
                return;
        }
    }
    assert(false);
}

/**
 * Calculates the data required for a cumulative plot of the data for all optimal policies.
 */
void calculateCumulativeOptimalPolicyData(in int n, in int portfolioSize)
{
    const path = DirectoryType.INFORMATION ~ dirSeparator ~ "all_optimal_policies" ~ dirSeparator;
    const fileName = "n=" ~ n.text() ~ "_" ~ "k=" ~ portfolioSize.text();
    const filePath = path ~ fileName;
    if(!exists(filePath))
    {
        return;
    }
    auto policyData = readText(filePath).split("\n")[1 .. $ - 1]
        .map!(line => line.split(" "))().array();
    policyData.sort!((a, b) => a[0].to!(double) < b[0].to!(double))();

    const outputPath = DirectoryType.INFORMATION ~ dirSeparator ~ "cumulative_optimal_policies" ~ dirSeparator;
    mkdirRecurse(outputPath);
    auto outputFile = File(outputPath ~ fileName, "w");
    outputFile.writeln("relative_run_time relative_amount portfolio");
    foreach(index, policyEntry; policyData)
    {
        outputFile.write(policyEntry[0].to!(double) / (n ^^ 2), " ", (index + 1.0) / policyData.length, " ",
            policyEntry[1 .. $].join());
    }
}


// ****************
// ** Evaluation **
// ****************

/** 
 * Evaluate multiple tests at once, given as an array of test cases.
 * Requires results from runs to be present. If not, no evaluation is generated.
 */
void evaluateDifferentRLSTests(in TestCase[] testCases)
{
    foreach(testCase; testCases)
    {
        evaluateRuns(testCase.n, testCase.portfolioSize, testCase.portfolioName, testCase.pickerName);
    }
}

/** 
 * Evaluates the data produced by runs. Just returns if no such data is present. Calculates the average of this data.
 */
void evaluateRuns(in int n, in int portfolioSize, in PortfolioName portfolioName, in PickerName pickerName)
{
    auto experimentsFileData = getFileData(DirectoryType.EXPERIMENTS, n, portfolioSize, portfolioName, pickerName);
    auto evaluationsFileData = getFileData(DirectoryType.EVALUATION, n, portfolioSize, portfolioName, pickerName);
    if(!exists(experimentsFileData.filePath))
    {
        return;
    }
    auto experimentsFile = File(experimentsFileData.filePath);

    auto lines = experimentsFile.byLineCopy()
                                .array()
                                .map!(l => l.strip.to!(int))()
                                .array()
                                .sort()
                                .array();
    auto evaluationFile = File(evaluationsFileData.filePath, "w");
    evaluationFile.writeln("avg lq med uq");
    evaluationFile.writeln
    (
        lines.sum() / lines.length, " ",
        lines[cast(int) (0.25 * n)], " ",
        lines[cast(int) (0.50 * n)], " ",
        lines[cast(int) (0.75 * n)],
    );
}


// ***********
// ** Tests **
// ***********

/** 
 * Run multiple tests at once, given as an array of test cases.
 */
void runDifferentRLSTests(in TestCase[] testCases, in int numberOfRuns)
{
    foreach(testCase; testCases)
    {
        runRLSTests(testCase.n, testCase.portfolioSize, testCase.portfolioName, testCase.pickerName, numberOfRuns);
    }
}

/** 
 * Runs the provided RLS variant with the given portfolio and policy for the given number of times.
 */
void runRLSTests(in int n, in int portfolioSize, in PortfolioName portfolioName, in PickerName pickerName,
    in int numberOfRuns)
{
    // Choose the correct portfolio and mutation picker.
    const portfolio = calculatePortfolio(n, portfolioSize, portfolioName);
    // Abort if the portfolio contains invalid (i.e., too large) mutation rates.
    if(portfolio[0] > n)
    {
        return;
    }

    MutationPicker mutationPicker = ()
        {
            with(PickerName)
            {
                final switch(pickerName)
                {
                    case RANDOM:
                        return getUniformPicker(portfolio);
                    case OPTIMAL:
                        return getOptimalPicker(portfolio, n);
                }
            }
        }();
    
    // Set up the file.
    auto experimentsFileData = getFileData(DirectoryType.EXPERIMENTS, n, portfolioSize, portfolioName, pickerName);
    auto file = File(experimentsFileData.filePath, "w");
    file.close();

    // Run tests in parallel.
    foreach(_; iota(numberOfRuns).parallel())
    {
        auto numberOfIterations = rlsWithSchedule(n, mutationPicker);
        synchronized
        {
            file.open(experimentsFileData.filePath, "a");
            file.writeln(numberOfIterations);
            file.close();
        }
    }
}