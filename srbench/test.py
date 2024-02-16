import numpy as np
import tensorflow as tf
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os
tf.enable_eager_execution()

from lang.evolution import GeneticAlgorithm, AdaptiveGA, umad_creator
from lang.selection import Lexicase, WeightedLexicase, Tournament
from lang.variation import UMAD, Variation
from lang.interpreter import SRInterpreter, ErrorFunction 
from lang.instruction_set import InstructionSet
from lang.instructions import normalint, scalar_set, get_inputs, korns_instr_scalar, scalar_uniform_erc, limit_tensor, scalar_normal_erc, vlada_instr_scalar, koza_instr_scalar
from lang.data import korns12, pagie1, nguyen7, vladislavleva4, keijzer6
from read_file import read_file 
from gplearn_wrapper import SymbolicRegressor, mse


parser = argparse.ArgumentParser(description='Genetic Algorithm')
parser.add_argument("i", type=int)
parser.add_argument("inherit_depth", nargs="?")
parser.add_argument('--population_size', type=int, default=1000, help='Size of the population')
parser.add_argument('--max_generations', type=int, default=300, help='Number of generations')
parser.add_argument('--genome_size', type=int, nargs="+", default=[1,50], help='Limits of initialized program size')
parser.add_argument('--problem', type=str, default="korns12", help='Problem to solve')
parser.add_argument('--selector', type=str, default="epsilon lexicase", help='selection method')
parser.add_argument('--wlexicase_weight', type=float, default=1, help="standard deviation of weighted lexicase selection")
parser.add_argument('--wlexicase_distribution', type=str, default="normal", help="distribution of weighted lexicase selection")
parser.add_argument("--path_override", type=str, default=None, help="override default write location of json logs")
parser.add_argument("--path_append", type=str, default='', help="append to default write location of json logs")
parser.add_argument("--scale_xy", type=lambda x:x=="True", default=True, help="Whether to apply standard scaler to x and y")
parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to use from dataset. Subsamples larger datasets.")
parser.add_argument("--target_feature_noise", type=float, default=0.0, help="Gaussian noise to add to target and features")
parser.add_argument("--hpo", action="store_true")
parser.add_argument("--black_box", action="store_true")
parser.add_argument("--gplearn", action="store_true")
parser.add_argument("--tournament_size", type=int, default=20, help="size of tournaments for tournament selection")
parser.add_argument("--path_prepend", type=str, default='', help="prepend to path")


args = parser.parse_args()


rng = np.random.default_rng()
TRAIN_TEST_SPLIT = [0.75,0.25]



if args.black_box:
    features, labels, feature_names = read_file(f"{args.problem}/{args.problem}.tsv.gz")
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=TRAIN_TEST_SPLIT[0],
                                                    test_size=TRAIN_TEST_SPLIT[1])
else:
    problem = {"korns12":korns12, 
            "pagie1":pagie1, 
            "nguyen7":nguyen7,
            "vladislavleva4":vladislavleva4,
            "keijzer6":keijzer6}[args.problem]
    X_train, y_train = problem.train_data()
    X_test, y_test = problem.test_data()
    

# scale and normalize the data
if args.scale_xy:
    sc_X = StandardScaler() 
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test

if args.scale_xy:
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
else:
    y_train_scaled = y_train

rng = np.random.default_rng()

# add noise to the target
if args.target_feature_noise > 0:
    noise = args.target_feature_noise
    y_train_scaled += rng.normal(0, 
                noise*np.sqrt(np.mean(np.square(y_train_scaled))),
                size=len(y_train_scaled))

    X_train_scaled = np.array([x 
        + rng.normal(0, noise*np.sqrt(np.mean(np.square(x))),
                            size=len(x))
                                for x in X_train_scaled.T]).T

train_inputs, train_targets,unscaled_train_targets = X_train_scaled.astype(np.float32), y_train_scaled.astype(np.float32),y_train.astype(np.float32)
test_inputs, test_targets = X_test_scaled.astype(np.float32), y_test.astype(np.float32)

# instr = korns_instr_scalar + get_inputs(train_inputs.shape[1]) + [scalar_normal_erc()]
instr = koza_instr_scalar + get_inputs(train_inputs.shape[1]) + [scalar_uniform_erc(low=0,high=1)]
# instr = koza_instr_scalar + [get_inputs(5)[0], get_inputs(5)[-1]] + [scalar_uniform_erc(low=0,high=1)]
# instr = vlada_instr_scalar + get_inputs(train_inputs.shape[1]) + [scalar_uniform_erc(low=-5,high=5)]
instruction_set = InstructionSet(instr)

interpreter = SRInterpreter(initial_state = {"Scalar": [], "Vector": [], "Matrix": []})# error_fn = lambda x,y:tf.math.abs(x-y) 
error_function = ErrorFunction(interpreter,train_inputs,train_targets)

selectors = {"epsilon lexicase": lambda: Lexicase(epsilon=True),
            "weighted lexicase": lambda: WeightedLexicase(std=np.array(args.wlexicase_weight).reshape([1,1,1]), distribution=args.wlexicase_distribution),
            "tournament": lambda:Tournament(args.tournament_size)}

selector = selectors[args.selector]()
# selector = Lexicase(epsilon=True) if args.selector=="epsilon lexicase" else WeightedLexicase(std=np.array(args.wlexicase_weight).reshape([1,1,1]), distribution=args.wlexicase_distribution)

umad = UMAD(instruction_set, maxlen=args.genome_size[1])
variation = Variation([umad])

# if args.gplearn:
#     ga_maker = lambda sel,std,tour, gen: SymbolicRegressor(
#                         tournament_size=tour,
#                         init_depth=(2, 6),
#                         init_method='half and half',
#                         metric=mse,
#                         parsimony_coefficient=0.001,
#                         p_crossover=0.9,
#                         p_subtree_mutation=0.01, 
#                         p_hoist_mutation=0.01, 
#                         p_point_mutation=0.01, 
#                         p_point_replace=0.05,
#                         max_samples=1.0,
#                         selector= sel,
#                         std=std,
#                         function_set= ('add', 'sub', 'mul', 'div', 'log',
#                                         'sqrt', 'sin','cos'),
#                         population_size=args.population_size,
#                         generations=gen,
#                         verbose=True,)
#     ga = ga_maker(args.selector, args.wlexicase_weight, args.tournament_size, 300)
#     hpo_idx=-1
#     hpo_scores = -1
#     if args.hpo:
#         X_train, X_test, y_train, y_test = train_test_split(train_inputs, train_targets,
#                                                         train_size=0.75,
#                                                         test_size=0.25)
#         if args.selector == "epsilon lexicase":
#             ga = [ga_maker(args.selector, args.wlexicase_weight, args.tournament_size, 100) for _ in range(3)]
#         elif args.selector == "weighted lexicase":
#             ga = [ga_maker(args.selector, s, args.tournament_size, 100) for s in [0.5,1.,2.]]
#         elif args.selector == "tournament":
#             ga = [ga_maker(args.selector, args.wlexicase_weight, t, 100) for t in [5,10,20]]
#         else:
#             raise NotImplementedError

#         for i,g in enumerate(ga):
#             print(f"====================== HPO Search {i} ======================")
#             g.fit(X_train, y_train)
#         hpo_scores = [g.score(X_test,y_test) for g in ga] 
#         print("HPO RESULTS")
#         print(hpo_scores)
#         hpo_idx = max(range(len(hpo_scores)),key=lambda x:hpo_scores[x]["R^2"])
#         print(f"Chosen index: {hpo_idx}")
#         print("===========================================================")
#         print("\n")
#         print("====================== TRAINING LOOP ======================", flush=True)
#         ga=ga[hpo_idx]
# else:
#     ga = GeneticAlgorithm(error_function, instruction_set, selector, variation, args.population_size, args.max_generations, args.genome_size, eager=True)

#     hpo_idx=-1
#     hpo_scores = -1
#     if args.hpo:
#         X_train, X_test, y_train, y_test = train_test_split(train_inputs, train_targets,
#                                                         train_size=0.75,
#                                                         test_size=0.25)
#         if args.selector == "epsilon lexicase":
#             ga = [GeneticAlgorithm(error_function, instruction_set, selector, variation, args.population_size, int(args.max_generations/3), args.genome_size, eager=True) for _ in range(3)]
#         elif args.selector == "weighted lexicase":
#             selectors = [WeightedLexicase(std=np.array(std).reshape([1,1,1]), distribution=args.wlexicase_distribution) for std in [0.5,1,2]]
#             ga = [GeneticAlgorithm(error_function, instruction_set, selector, variation, args.population_size, int(args.max_generations/3), args.genome_size, eager=True) for s in selectors]
#         elif args.selector == "tournament":
#             selectors = [Tournament(s) for s in [5,10,20]]
#             ga = [GeneticAlgorithm(error_function, instruction_set, selector, variation, args.population_size, int(args.max_generations/3), args.genome_size, eager=True) for s in selectors]
#         else:
#             raise NotImplementedError


#         for i,g in enumerate(ga):
#             print(f"====================== HPO Search {i} ======================")
#             g.fit(X_train, y_train)
#         hpo_scores = [g.score(X_test,y_test) for g in ga] 
#         print("HPO RESULTS")
#         print(hpo_scores)
#         hpo_idx = max(range(len(hpo_scores)),key=lambda x:hpo_scores[x]["R^2"])
#         print(f"Chosen index: {hpo_idx}")
#         print("===========================================================")
#         print("\n")
#         print("====================== TRAINING LOOP ======================", flush=True)
#         ga=ga[hpo_idx]

# # from sklearn.utils.estimator_checks import check_estimator 
# # print(check_estimator(ga))

# t = time.time()

# if args.gplearn:
#     if args.hpo:
#         ga.set_params(generations=400, warm_start=True)
#         ga.fit(train_inputs, train_targets)
#     else:
#         ga.fit(train_inputs,train_targets)
#     ga_runtime = time.time()-t
# else:
#     if args.hpo:
#         ga.resume(args.max_generations, inputs=train_inputs, targets=train_targets)
#     else:
#         ga.fit(train_inputs,train_targets)
#     ga_runtime = time.time()-t

creator = umad_creator(error_function, instruction_set)
ga = AdaptiveGA(creator, error_function, args.population_size, int(args.max_generations/3), args.genome_size, eager=True)

ga.fit(train_inputs,train_targets)

test_error = ga.score(test_inputs,test_targets, scaler=sc_y if args.scale_xy else None)
train_error = ga.score(train_inputs,unscaled_train_targets, scaler=sc_y if args.scale_xy else None)
final_symbolic = None if args.gplearn else ga.best_seen.symbolic
complexity = ga.complexity()

print(f"\nFinal test R^2: {test_error}")
print(f"Final program: {final_symbolic}")
print(f"HPO index: {hpo_idx}]")
print(f"HPO scores: {hpo_scores}]\n")
print(ga.run_details_ if args.gplearn else ga.logs)

def coerce(d):
    if isinstance(d,dict):
        for k,v in d.items():
            if isinstance(v,dict):
                d[k]=coerce(v)
            elif isinstance(v, np.ndarray):
                d[k]=v.tolist()
            elif isinstance(v,list) or isinstance(v, tuple):
                d[k] = [coerce(_k) for _k in v]
    return d

if args.path_override != "none":
    out_dir = args.path_override or "results/"
    out_dir+=("" if args.path_prepend=="none" else args.path_prepend+"/")
    out_dir+=f"{args.problem}-{args.selector.replace(' ','_')}"
    out_dir+=("" if args.path_append=="none" else args.path_append)+"/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir+f"{args.i}.json"


    with open(out_file, 'w') as f:
        d = {"final symbolic": final_symbolic,
                "test error":test_error,
                "train error":train_error,
                "ga runtime":ga_runtime,
                "arguments":args.__dict__,
                "complexity":complexity,
                "hpo": [hpo_idx, hpo_scores],
                **(ga.run_details_ if args.gplearn else ga.logs)}
        json.dump(coerce(d), f)







