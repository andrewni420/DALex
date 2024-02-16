# DALex in Code Building Genetic Programming
This code is derived from [Code Building Genetic Programming](https://github.com/erp12/CodeBuildingGeneticProgramming-ProtoType) by Eddie Pantridge and [Plexicase Selection in Code Building Genetic Programming](https://github.com/ld-ing/plexicase/tree/main/CBGP) by Li Ding, which also builds on CBGP.

## Experiments
The experiments are tested with Python3.8.

To run experiments with lexicase selection:

```
cd CBGP
python3 run_lexicase.py exp_id downsample_rate
```

`exp_id` is composed of two parts, where

```
problem_id = exp_id // 100
    exp_id = exp_id % 100
```

So a full run of experiments will use `exp_id` from 0 to 699, which consists of 7 PSB problems with 100 trials for each problem.

`downsample_rate` is a float number from 0-1, which uses random downsampling on the training set during each generation. In this paper, we use a `downsample_rate` of 1 and 0.25.

To run experiments with plexicase selection:

```
cd CBGP
python3 run_lexiprob.py exp_id downsample_rate alpha
```

`alpha` is a float number from 0-inf, which is a hyperparameter we introduced to tune the shape of probability distribution of selection. Lower `alpha` gives more uniform distributions, and higher values gives more weights on elite solutions.

To run experiments with DALex:

```
cd CBGP
python3 run_dalex.py exp_id downsample_rate distribution std
```

where `distribution` controls the distribution from which importance weights are sampled, and can take on the values "normal", "uniform", or "range" as described in the paper. The default is "normal". 

`std` controls the standard deviation of the sampled importance weights, and is set to a default of 200 as recommended in the paper. Higher `std` results in more lexicase-like selection dynamics, and lower `std` in more relaxed selection dynamics.  
