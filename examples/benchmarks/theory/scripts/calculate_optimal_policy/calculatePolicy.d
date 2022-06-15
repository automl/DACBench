module app;

public import data_management;

import std.conv;

void main(string[] args)
{
    const int n = args[1].to!(int);
    const int[] portfolio = args[2 .. $].to!(int[]);

    writeln(portfolio.determineOptimalBreakingPoints(n)[1 .. $]);
}