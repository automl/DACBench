import sys
import os
import subprocess

def run_cmd(cmd):
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    output = p.stdout.decode("utf-8")
    return output, p.returncode


def calculate_optimal_policy(n: int, portfolio: [int], scriptDir: str):
    # call Martin's D code to get the optimal policy
    portfolio = sorted(portfolio, reverse=True)
    cmd = f"{scriptDir}/calculatePolicy {n} {' '.join([str(x) for x in portfolio])}"
    output, rc = run_cmd(cmd)
    assert rc == 0, f"ERROR: fail to run command: {cmd}"
    assert n > 0, f"ERROR: problem size must be a positive integer"
    breakPoints = [
        int(s.strip()) for s in output.replace("[", "").replace("]", "").split(",")
    ]
    assert len(breakPoints) == len(portfolio)

    # remove radius that are not used (due to duplicated breaking points)
    newPort = []
    newBreakPoints = []
    for i in range(len(breakPoints)):
        skip = False
        if i > 0:
            if breakPoints[i] == breakPoints[i - 1]:
                skip = True
            else:
                assert breakPoints[i] > breakPoints[i - 1]
        if skip:
            continue
        newPort.append(portfolio[i])
        newBreakPoints.append(breakPoints[i])

    # parse the optimal policy to an array of radiuses ([0..n-1])
    policy = []
    previousBP = -1
    for i in range(len(newPort)):
        policy.extend([newPort[i]] * (newBreakPoints[i] - previousBP))
        previousBP = newBreakPoints[i]

    assert len(policy) == n
    return policy


def main():
    assert (
        len(sys.argv) == 3
    ), "Usage: python calculate_optimal_policy.py <n> <portfolio>. \nExample: python calculate_optimal_policy.py 50 1,17,33"
    n = int(sys.argv[1])
    portfolio = [int(s.strip()) for s in sys.argv[2].split(",")]
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    p = calculate_optimal_policy(n, portfolio, scriptDir)
    print(" ".join([str(x) for x in p]))


if __name__ == "__main__":
    main()
