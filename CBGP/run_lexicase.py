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
from push4.gp.selection import Lexicase
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
parser.add_argument("inherit_depth", nargs="?")
parser.add_argument('--track_fitnesses', type=bool, default=None, help='Whether to compute selection probabilities')
parser.add_argument('--alpha', type=int, default=1, help='Probability manipulation')
parser.add_argument('--sample_size', type=int, default=1, help='Number of individuals to sample to compute selection probabilities')
parser.add_argument('--perturb', type=str, default=None, nargs="+", help='Error vector perturbation parameters')
args = parser.parse_args()

def run(problem: Problem, downsample_rate=1., inherit_depth=0, sample_size=0):
    # The spawner which will generate random genes and genomes.
    spawner = Spawner(problem.soup())

    # The parent selector.
    selector = Lexicase(epsilon=False, alpha=args.alpha)

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
        inherit_depth=inherit_depth,
        fitnesses = [] if args.track_fitnesses else None,
        sample_size = sample_size,
        perturb=perturb
    )

    # simplifier = GenomeSimplifier(problem.train_error, problem.output_type)

    best = evo.run(problem.output_type)
    fn_name = problem.name.replace("-", "_")
    print(best.program.to_def(fn_name, problem.arg_names), flush=True)
    print("", flush=True)

    # simp_best = simplifier.simplify(best)
    # print(simp_best.program.to_def(fn_name, problem.arg_names))
    # print()

    generalization_error_vec = problem.test_error(best.program).round(5)
    print(generalization_error_vec, flush=True)
    print("Final Test Error:", generalization_error_vec.sum(), flush=True)

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
        inherit_depth = int(sys.argv[3])
    except:
        inherit_depth = 0

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
        # "SLB": StringLengthBackwards()
    }
    PROBLEM_NAMES = list(PROBLEMS.keys())
    problem_name = PROBLEM_NAMES[problem_id]
    problem = PROBLEMS[problem_name]
    # print(f"PROBLEM NAME: {problem_name}")

    out_dir = "results/ds_{}/{}-lexicase-inh{}{}-a{}".format(
        downsample_rate, problem_name,inherit_depth,"-track-fitness" if args.track_fitnesses else "",args.alpha)
    os.makedirs(out_dir, exist_ok=True)

    out_file = out_dir+'/exp_{}.json'.format(exp_id)
    if not os.path.exists(out_file):
        test_err, logs,fitnesses = run(problem, downsample_rate, inherit_depth=inherit_depth, sample_size = args.sample_size)
        results = logs
        results['test_err'] = float(test_err)
        results['problem'] = problem_name
        results['method'] = 'lexicase'

        with open(out_file, 'w') as f:
            json.dump(results, f)

        if fitnesses is not None:
            fitness_file = out_dir+'/fitnesses_{}.json'.format(exp_id)
            with open(fitness_file, 'w') as f:
                json.dump(fitnesses, f) 

    
