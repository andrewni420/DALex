import numpy as np
import argparse
import json
import time
import os
import re 
from scipy.stats import wilcoxon, bootstrap
import matplotlib.pyplot as plt
from itertools import chain

from seeds import SEEDS

parser = argparse.ArgumentParser(description='Genetic Algorithm')
parser.add_argument('--dir', type=str, default=[], nargs="+", help='result directories')
parser.add_argument('--n', type=int,default=10, help='number of files')
parser.add_argument('--prepend', type=str,default="none", help='prepend to directory')
parser.add_argument('--selector', type=str,default="weighted_lexicase", help='selector to use')

args = parser.parse_args()

seeds = SEEDS[:args.n]

sr_dirs = [  '586_fri_c3_1000_25','192_vineyard',  '519_vinnie', '564_fried',  '583_fri_c1_1000_50',"344_mv",
            '620_fri_c1_1000_25', '618_fri_c3_1000_50', '485_analcatdata_vehicle',
              '606_fri_c2_1000_10', '603_fri_c0_250_50','197_cpu_act',
             '602_fri_c3_250_10', '593_fri_c1_1000_10', '607_fri_c4_1000_50',
            '706_sleuth_case1202', '626_fri_c2_500_50', '644_fri_c4_250_25',
             ]#'344_mv' '505_tecator','631_fri_c1_500_5'

# sr_dirs = ["344_mv", "564_fried"]

sr_lengths = {"586_fri_c3_1000_25": 1000, "192_vineyard": 52, "197_cpu_act": 8192, "519_vinnie": 380, "564_fried": 40768, 
                "344_mv": 40768, "583_fri_c1_1000_50": 1000, "620_fri_c1_1000_25": 1000, "618_fri_c3_1000_50": 1000, 
                "485_analcatdata_vehicle": 48, "606_fri_c2_1000_10": 1000, "603_fri_c0_250_50": 250, "602_fri_c3_250_10": 250, 
                "593_fri_c1_1000_10": 1000, "607_fri_c4_1000_50": 1000, "706_sleuth_case1202": 93, "626_fri_c2_500_50": 500, 
                "644_fri_c4_250_25": 250}

# methods = ["epsilon-lexicase", "weighted-lexicase", "tournament"]
#strogatz_shearflow1 strogatz_bacres2
#618_fri_c3_1000_50 583_fri_c1_1000_50

selectors = ["weighted_lexicase_uniform_hpo9", "weighted_lexicase_hpo9", "weighted_lexicase", "epsilon_lexicase", "tournament-noparsimony", "tournament"]
names = ["weighted_lexicase", "weighted_lexicase", "weighted_lexicase", "epsilon_lexicase", "gplearn"]
# selectors = ["weighted_lexicase_100k", "epsilon_lexicase_100k"]
# names = ["weighted_lexicase", "epsilon_lexicase"]
# selectors = ["weighted_lexicase0.5", "weighted_lexicase1.0", "weighted_lexicase2.0", "weighted_lexicase3.0", "weighted_lexicase5.0", "weighted_lexicase7.0", "weighted_lexicase"]

selectors = selectors[2:]
names = names[2:]

def get_metrics(metrics, problem, selector_idx):
    if not isinstance(metrics,list):
        metrics = [metrics]
    data = []
    for seed in seeds:
        f = open("/home/ani24/srbench/experiment/results_blackbox/" + ("" if args.prepend=="none" else args.prepend+"/") + f"{selectors[selector_idx]}/{problem}/{problem}_{names[selector_idx]}_{seed}.json","r")
        data.append(json.loads(f.read()))
    return [{m:i[m] for m in metrics} for i in data]

def median_r2(problem,selector_idx):
    r2 = [i["r2_test"] for i in get_metrics(["r2_test"],problem,selector_idx)]
    return np.median(r2)

def median_time(problem,selector_idx):
    r2 = [i["process_time"] for i in get_metrics(["process_time"],problem,selector_idx)]
    return r2

def median_size(problem,selector_idx):
    r2 = [i["model_size"] for i in get_metrics(["model_size"],problem,selector_idx)]
    return np.median(r2)

def processtest(test_stats, fn=np.median):
    N_SAMPLES=10000
    res = fn(test_stats)
    bootstrap_res = bootstrap((test_stats,),fn, n_resamples=N_SAMPLES)
    return res,bootstrap_res

def plot_result(res):
    conf = [r[1].confidence_interval for r in res]

    sel = ["DALex (std=3)", "Epsilon Lexicase", "Tournament"]
    med = [r[0] for r in res]
    from scipy.stats import median_test
    
    err = [[m-l,h-m] for m,(l,h) in zip(med,conf)]
    print(f"MEDIANS {med}")

    fig, ax = plt.subplots()
    ax.bar(sel,med)
    ax.errorbar(sel,med, yerr=np.array(err).transpose(), fmt="none", color="r", capsize=5)
    # ax.set_title('Bootstrapped Median 10-run-median R² on Black-Box Regression')
    ax.set_ylabel('Median R²', fontsize=20)
    ax.set_xlabel("Selection Method", fontsize=20)
    ax.set_ylim([0,1.0])
    plt.xticks(rotation=10, fontsize=17)
    plt.savefig("srbench.pdf",format="pdf")
   
    plt.show()

def plot_time(single_time, axes, label, fn=np.median, color="green"):
    x = np.unique(list(sr_lengths.values()))
    labels = [[s for s in sr_dirs if sr_lengths[s]==i] for i in x]
    y_total = [np.array([single_time[l] for l in item]).flatten() for item in labels]
    y_median = [fn(item) for item in y_total]
    y_ci = [bootstrap((item,),fn).confidence_interval for item in y_total]
    y_error = [[m-l,h-m] for m,(l,h) in zip(y_median,y_ci)]
    y_median = np.array(y_median)/3600
    y_error = np.array(y_error)/3600

    
    y_ = [[fn(single_time[l])/3600 for l in item] for item in labels]
    x_ = [[np.log(i)/np.log(10)]*len(j) for i,j in zip(x,y_)]

    axes[0].scatter(list(chain.from_iterable(x_)),list(chain.from_iterable(y_)), label=label)
    axes[1].scatter(list(chain.from_iterable(x_)),list(chain.from_iterable(y_)), label=label)
    # axes[0].plot(np.log(x)/np.log(10), y_median, label=label, color=color)
    # axes[0].errorbar(np.log(x)/np.log(10), y_median, yerr = np.array(y_error).transpose(), fmt="o", color=color)

    # axes[0].plot(np.log(x)/np.log(10), y_median, label=label, color=color)
    # axes[0].errorbar(np.log(x)/np.log(10), y_median, yerr = np.array(y_error).transpose(), fmt="o", color=color)
    
    axes[0].set_ylim(0, 1.5)
    axes[1].set_ylim(8,10)

    axes[1].spines['bottom'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].xaxis.tick_top()
    axes[1].tick_params(labeltop=False)  # don't put tick labels at the top
    axes[0].xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axes[1].transAxes, color='k', clip_on=False)
    axes[1].plot((-d, +d), (-d*3, +d*3), **kwargs)        # top-left diagonal
    axes[1].plot((1 - d, 1 + d), (-d*3, +d*3), **kwargs)  # top-right diagonal

    
    kwargs.update(transform=axes[0].transAxes)  # switch to the bottom axes
    axes[0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axes[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


times = {}
for s in range(len(selectors)):
    selector_time={}
    for problem in sr_dirs:
        selector_time[problem]=median_time(problem,s)
    times[selectors[s]]=selector_time


def plot_size(res):
    conf = [r[1].confidence_interval for r in res]

    sel = ["DALex (std=3)", "Epsilon Lexicase", "Tournament"]
    med = [r[0] for r in res]
    from scipy.stats import median_test
    
    err = [[m-l,h-m] for m,(l,h) in zip(med,conf)]
    print(f"MEDIANS {med}")

    fig, ax = plt.subplots()
    ax.bar(sel,med)
    ax.errorbar(sel,med, yerr=np.array(err).transpose(), fmt="none", color="r", capsize=5)
    # ax.set_title('Bootstrapped Median 10-run-median R² on Black-Box Regression')
    ax.set_ylabel('Median Model Size', fontsize=20)
    ax.set_xlabel("Selection Method", fontsize=20)
    # ax.set_ylim([0,1.0])
    plt.xticks(rotation=10, fontsize=17)
    plt.savefig("srbench size.pdf",format="pdf")
    
    plt.show()

plt.rcParams.update({'font.size': 17})
plt.rcParams['figure.constrained_layout.use'] = True

for k in sr_lengths.keys():
    sr_lengths[k]=min(10000,0.7*sr_lengths[k])

colors = ["gold", "darkcyan", "maroon"]
f, (ax2,ax) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 3]))
plt_names = ["DALex (std=3)", "Epsilon Lexicase", "Tournament"]
for k,n,c in zip(times.keys(),plt_names,colors):
    plot_time(times[k], (ax,ax2), n,color=c)

mv_344 = [np.log(sr_lengths["344_mv"])/np.log(10), np.median(times["epsilon_lexicase"]["344_mv"])/3600]
ax2.text(mv_344[0],mv_344[1],'   344_mv   ', horizontalalignment='right')

ax2.legend()
f.supylabel("Training Time (h)",fontsize=20)
plt.xlabel("Dataset size",fontsize=20)
plt.xticks([2,3,4],[100,1000,10000],fontsize=17)

plt.savefig("srbench runtime.pdf",format="pdf")
# plt.savefig("srbench runtime2.pdf",format="pdf")
plt.show()

import sys 
# sys.exit(0)


scores = []
for s in range(len(selectors)):
    selector_score=[]
    for problem in sr_dirs:
        selector_score.append(median_size(problem,s))
    scores.append(selector_score)

def processsize(test_stats, fn=np.median):
    # print(test_stats)
    N_SAMPLES=10000
    res = fn(test_stats)
    bootstrap_res = bootstrap((test_stats,),fn, n_resamples=N_SAMPLES, method="basic")
    return res,bootstrap_res

# print(processtest(scores[0]))
# print(sorted([49.0, 42.0, 7.5, 45.0, 50.0, 43.0, 60.0, 45.0, 43.5, 49.5, 47.0, 20.0, 47.0, 56.0, 54.0, 43.5, 51.5, 48.0]))
# print(bootstrap(([7.5, 20.0, 42.0, 43.0, 43.5, 43.5, 45.0, 45.0, 47.0, 47.0, 48.0, 49.0, 49.5, 50.0, 51.5, 54.0, 56.0, 60.0],), np.median, n_resamples=10000, method="basic"))
# print(np.median([49.0, 42.0, 7.5, 45.0, 50.0, 43.0, 60.0, 45.0, 43.5, 49.5, 47.0, 20.0, 47.0, 56.0, 54.0, 43.5, 51.5, 48.0]))
res = [processsize(t,fn=np.median) for t in scores]

plot_size(res)


scores = []
for s in range(len(selectors)):
    selector_score=[]
    for problem in sr_dirs:
        selector_score.append(median_r2(problem,s))
    scores.append(selector_score)

res = [processtest(t,fn=np.median) for t in scores]

plot_result(res)

