from __future__ import annotations

import sys
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import random
import json
import argparse

from push4.gp.evolution import GeneticAlgorithm, Lexiprob
from push4.gp.selection import DALex
from push4.gp.simplification import GenomeSimplifier
from push4.gp.soup import Soup, CoreSoup, GeneToken
from push4.gp.spawn import Spawner, genome_to_push_code
from push4.gp.variation import size_neutral_umad, VariationSet
from push4.lang.dag import Dag
from push4.lang.expr import Input, Function, Method, Constant
from push4.lang.hof import LocalInput, MapExpr
from push4.lang.push import Push
from push4.lang.reify import RetToElementType, ArgsToElementType
from push4.library.io import do_print, _pass_do, print_do
from push4.library.op import add, _max_numeric, sum_, div, max_
from push4.library.str import String
from push4.library.collections import len_, in_
from push4.utils import damerau_levenshtein_distance

from problems import *

parser = argparse.ArgumentParser(description='Error grapher')
parser.add_argument("problem")
parser.add_argument("downsample_rate")
parser.add_argument("distribution")
parser.add_argument("std")
parser.add_argument("inherit_depth", nargs="?")
parser.add_argument('--alpha', type=int, default=1, help='Probability manipulation')
parser.add_argument('--track_fitnesses', type=bool, default=None, help='Whether to compute selection probabilities')
parser.add_argument('--sample_size', type=int, default=1, help='Number of individuals to sample to compute selection probabilities')
parser.add_argument('--discount', type=float, default=0.5, help='Discount factor applied to inherited errors')
parser.add_argument('--std_to_scale', type=float, default=1, help='Conversion between std of sampled weights and distance between means of inherited error weights')
parser.add_argument('--perturb', type=str, default=None, nargs="+", help='Error vector perturbation parameters')

args = parser.parse_args()

TRACK_FITNESSES = args.track_fitnesses

def construct_parameters(std,inherit_depth):
    params=std.split("-")
    try:
        std=float(params[0])
    except:
        std=20.
    try:
        sts=float(params[1])
    except:
        sts=2
    try:
        discount = float(params[2])
    except:
        discount=0.5

    offsets = np.power(discount,np.arange(inherit_depth)) * (std * sts)
    offsets = np.insert(np.cumsum(offsets),0,0)
    offsets = offsets-offsets.mean()
    parameters = [offsets[i:i+2] for i in range(len(offsets)-1)]
    shape = [1,inherit_depth,1]
    std = np.array([abs(p[1]-p[0])/2 for p in parameters]).reshape(shape)
    offsets = np.array([(p[0]+p[1])/2 for p in parameters]).reshape(shape)

    return std,offsets

def inherit_weights(std=20, inherit_depth=0, discount=0.5, sts=2):
    std = float(std)
    inherit_depth = int(inherit_depth)
    discount = float(discount)
    sts = float(sts)
    # print(f"ARANGE {np.arange(inherit_depth+1)}")
    # print(f"PREPAD {discount**np.arange(inherit_depth+1)*std*sts}")
    # print(f"PAD {np.pad(discount**np.arange(inherit_depth+1)*std*sts,((1,0),))}")
    init = np.cumsum(np.pad(discount**np.arange(inherit_depth+1)*std*sts,(1,0)))
    # print(f"INIT {init}")
    init-=np.mean(init)
    # print(f"INIT {init}")
    init=np.stack([init,np.roll(init,-1)])[:,:-1]
    # print(f"INIT {init}")
    loc = init.mean(axis=0)
    # print(f"LOC {loc}")
    scale = (init[1]-init[0])/sts
    # print(f"SCALE {scale}")
    # print(f"STD {std} INH {inherit_depth} DISC {discount} OUTPUT {np.reshape(scale,[1,scale.size,1]),np.reshape(loc,[1,loc.size,1])}")
    return np.reshape(scale,[1,scale.size,1]),np.reshape(loc,[1,loc.size,1])

def run(problem: Problem, downsample_rate=1., distribution="normal", std=1, inherit_depth=0, offset=0, sample_size=0, std_to_scale=1, discount=1):
    # The spawner which will generate random genes and genomes.
    spawner = Spawner(problem.soup())

    std,offset = inherit_weights(std=std, inherit_depth=inherit_depth,discount=discount,sts=std_to_scale)

    # The parent selector.
    selector = DALex(std=std, distribution=distribution, offset=offset, alpha=args.alpha)

    perturb = args.perturb
    if perturb is not None:
        perturb[-1]=float(perturb[-1])

    # The evolver
    evo = GeneticAlgorithm(
        downsampler = problem.sample_cases,
        error_function=problem.train_error,
        spawner=spawner,
        selector=selector,
        variation=VariationSet([
            (size_neutral_umad, 1.0),
        ]),
        population_size=1000,
        max_generations=300,
        initial_genome_size=(10, 60),
        downsample_rate=downsample_rate,
        inherit_depth = inherit_depth,
        fitnesses=[] if TRACK_FITNESSES else None,
        sample_size=  sample_size
    )

    # simplifier = GenomeSimplifier(problem.train_error, problem.output_type)

    best = evo.run(problem.output_type)
    fn_name = problem.name.replace("-", "_")
    print(best.program.to_def(fn_name, problem.arg_names))
    print()

    # simp_best = simplifier.simplify(best)
    # print(simp_best.program.to_def(fn_name, problem.arg_names))
    # print()

    generalization_error_vec = problem.test_error(best.program).round(5)
    print(generalization_error_vec)
    print("Final Test Error:", generalization_error_vec.sum())

    return generalization_error_vec.sum(), evo.logs, evo.fitnesses


if __name__ == "__main__":
    try:
        exp_id = int(sys.argv[1])
    except:
        exp_id = 0

    try:
        downsample_rate = float(sys.argv[2])
    except:
        downsample_rate = 1.

    try:
        distribution = str(sys.argv[3])
    except:
        distribution = "normal"

    try:
        inherit_depth = int(sys.argv[5])
        inherit_depth_name=sys.argv[5]
    except:
        inherit_depth = 0
        inherit_depth_name="0"

    try:
        std_name = sys.argv[4]
        std,offset = construct_parameters(sys.argv[4],inherit_depth+1)
    except:
        std = np.array(1.).reshape([1,1,1])
        offset = np.array(0.).reshape([1,1,1])
        std_name="1"

    
    problem_id = exp_id // 100
    exp_id = exp_id % 100
    np.random.seed(exp_id)
    random.seed(exp_id)

    PROBLEMS = {
        "csl": CompareStringLengths(),
        "median": Median(),
        "number-io": NumberIO(),
        "rswn": ReplaceSpaceWithNewline(),
        "smallest": Smallest(),
        "vector-average": VectorAverage(),
        "ntz": NegativeToZero(),
        "fuel-cost": FuelCost(),
        "fizz-buzz": FizzBuzz(),
        "middle-character": MiddleCharacter()
    }
    PROBLEM_NAMES = list(PROBLEMS.keys())
    problem_name = PROBLEM_NAMES[problem_id]
    problem = PROBLEMS[problem_name]

    out_dir = "results/ds_{}/{}-dalex-{}-{}-inh{}{}".format(
        downsample_rate, problem_name,distribution,std_name,inherit_depth_name,"-track-fitness" if args.track_fitnesses else "")
    os.makedirs(out_dir, exist_ok=True)

    out_file = out_dir+'/exp_{}.json'.format(exp_id)
    if not os.path.exists(out_file):
        test_err, logs, fitnesses = run(problem, downsample_rate=downsample_rate, std=std, distribution=distribution, inherit_depth = inherit_depth, offset=offset, sample_size=args.sample_size, discount=args.discount,std_to_scale=args.std_to_scale)
        results = logs
        results['test_err'] = float(test_err)
        results['problem'] = problem_name
        results['method'] = 'DALex'

        with open(out_file, 'w') as f:
            json.dump(results, f)

        if fitnesses is not None:
            fitness_file = out_dir+'/fitnesses_{}.json'.format(exp_id)
            with open(fitness_file, 'w') as f:
                json.dump(fitnesses, f) 
