import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

sns.set()

from matplotlib import rcParams

rcParams["font.size"] = "40"
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["figure.figsize"] = (16.0, 9.0)
rcParams["figure.frameon"] = True
rcParams["figure.edgecolor"] = "k"
rcParams["grid.color"] = "k"
rcParams["grid.linestyle"] = ":"
rcParams["grid.linewidth"] = 0.5
rcParams["axes.linewidth"] = 3
rcParams["axes.edgecolor"] = "k"
rcParams["axes.grid.which"] = "both"
rcParams["legend.frameon"] = "True"
rcParams["legend.framealpha"] = 1
rcParams["legend.fontsize"] = 30

rcParams["ytick.major.size"] = 32
rcParams["ytick.major.width"] = 6
rcParams["ytick.minor.size"] = 6
rcParams["ytick.minor.width"] = 1
rcParams["xtick.major.size"] = 32
rcParams["xtick.major.width"] = 6
rcParams["xtick.minor.size"] = 6
rcParams["xtick.minor.width"] = 1
rcParams["xtick.labelsize"] = 32
rcParams["ytick.labelsize"] = 32


def dir_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path to a file"
        )


parser = argparse.ArgumentParser(description="Script to plot LTO test data.")
parser.add_argument(
    "--log_path",
    type=dir_path,
    help="Path to the data file.",
    default=os.path.join("..", "examples", "10BBOB", "GallaghersGaussian21hi_LTO.json"),
)

args = parser.parse_args()
log_path = args.log_path

popsize = 10

data_run = {}

with open(log_path) as json_file:
    data_LTO = json.load(json_file)
generations = len(data_LTO["Average costs LTO"])
num_feval = generations * popsize

plt.tick_params(axis="x", which="minor")
plt.legend(loc=0, fontsize=25, ncol=2)
plt.xlabel("Num FEval", fontsize=50)
plt.ylabel("Step Size", fontsize=50)
plt.xticks(
    np.arange(start=1, stop=generations, step=generations // 5),
    [str(10)]
    + [
        str(gen * 10)
        for gen in np.arange(start=10, stop=generations, step=generations // 5)
    ],
)
plt.xticks()

plt.fill_between(
    list(np.arange(1, len(data_LTO["Sigma LTO"]) + 1)),
    np.subtract(data_LTO["Sigma LTO"], data_LTO["Std Sigma LTO"]),
    np.add(data_LTO["Sigma LTO"], data_LTO["Std Sigma LTO"]),
    color=sns.xkcd_rgb["magenta"],
    alpha=0.1,
)
plt.plot(
    list(np.arange(1, len(data_LTO["Sigma LTO"]) + 1)),
    data_LTO["Sigma LTO"],
    linewidth=4,
    label="LTO",
    color=sns.xkcd_rgb["magenta"],
)
plt.legend()
type = "StepSize"
output_path = os.path.join("..", "plots")
os.makedirs(output_path, exist_ok=True)
plot_file = "Plot_%s.pdf" % type
plt.savefig(os.path.join(output_path, plot_file), bbox_inches="tight")
plt.clf()


plt.tick_params(axis="x", which="minor")
plt.legend(loc=0, fontsize=25, ncol=2)
plt.xlabel("Num FEval", fontsize=50)
plt.ylabel("Objective Value", fontsize=50)
plt.xscale("log")
plt.xticks(
    np.arange(start=1, stop=generations, step=generations // 5),
    [str(10)]
    + [
        str(gen * 10)
        for gen in np.arange(start=10, stop=generations, step=generations // 5)
    ],
)

plt.fill_between(
    list(np.arange(1, len(data_LTO["Average costs LTO"]) + 1)),
    np.subtract(data_LTO["Average costs LTO"], data_LTO["Std costs LTO"]),
    np.add(data_LTO["Average costs LTO"], data_LTO["Std costs LTO"]),
    alpha=0.1,
    color=sns.xkcd_rgb["magenta"],
)
plt.plot(
    list(np.arange(1, len(data_LTO["Average costs LTO"]) + 1)),
    data_LTO["Average costs LTO"],
    linewidth=4,
    label="LTO",
    color=sns.xkcd_rgb["magenta"],
)

plt.legend()
type = "ObjectiveValue"
plot_file = "Plot_%s.pdf" % type
plt.savefig(os.path.join(output_path, plot_file), bbox_inches="tight")
plt.clf()
