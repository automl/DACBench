import math

with open("cmds.txt", "rt") as f:
    cmds = f.readlines()

name = "epsilon"
nSplits = 10
n = math.ceil(len(cmds)/nSplits)

startId = 0
for i in range(nSplits):
    endId = min(len(cmds), startId+n)
    lsCmds = cmds[startId:endId]
    startId += n
    cmdFile = f"cmds{i+1}.txt"
    with open(cmdFile, "wt") as f:
        f.write("".join(lsCmds))
    logFile = f"{name}-{i+1}.plog"
    pbsCmd = f"qsub -N {name}-{i+1} -v cmdFile={cmdFile},logFile={logFile} run1.pbs"
    print(pbsCmd)
