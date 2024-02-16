"""The :mod:`selection` module defines functions to select indices from error matrices."""
from operator import attrgetter

import numpy as np
from scipy.special import softmax


def lexicase(fitnesses, n=1, epsilon=False):
    errors = dict()
    for i,f in enumerate(fitnesses):
        error_vector_hash = hash(f.tobytes())
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append((i,f))
        else:
            errors[error_vector_hash]=[(i,f)]

    error_groups = [(v[0][1],[_i for _i,_f in v]) for _,v in errors.items()]

    error_matrix = np.array([g[0] for g in error_groups])
    
    inherit_depth, num_cases = error_matrix[0].shape
    popsize = len(error_groups)
    rng = np.random.default_rng()
    
    if isinstance(epsilon, bool):
        if epsilon:
            ep = np.median(np.abs(error_matrix - np.median(error_matrix,axis=0)),axis=0)
        else:
            ep = np.zeros([inherit_depth,num_cases])
    else:
        ep = epsilon

    selected = []

    for _ in range(n):
        candidates = range(popsize)
        for i in range(inherit_depth):
            if len(candidates) <= 1:
                break
            ordering = rng.permutation(num_cases)
            for case in ordering:
                if len(candidates) <= 1:
                    break
                errors_this_case = [error_matrix[ind][i][case] for ind in candidates]
                best_val_for_case = min(errors_this_case)+ep[i][case]
                candidates = [ind for ind in candidates if error_matrix[ind][i][case] <= best_val_for_case]
        selected.append(rng.choice(candidates))

    return [rng.choice(error_groups[i][1]) for i in selected]


def wlexicase(fitnesses, match_matrix=None, n=1, std=np.array(1).reshape([1,1,1]), offset=np.array(0).reshape([1,1,1]), distribution="normal"):
    std = np.array(std).astype(np.float32)
    offset=np.array(offset).astype(np.float32)
    std = std if std.ndim==3 else np.reshape(std,[1,std.shape[0] if std.ndim==1 else 1,1])
    offset = offset if offset.ndim==3 else np.reshape(offset,[1,offset.shape[0]if offset.ndim==1 else 1,1])
    errors = dict()
    matches = None if match_matrix is None else dict()
    for i,f in enumerate(fitnesses):
        error_vector_hash = hash(f.tobytes())
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append((i,f))
            if matches is not None:
                matches[error_vector_hash].append((i,match_matrix[i]))
        else:
            errors[error_vector_hash]=[(i,f)]
            if matches is not None:
                matches[error_vector_hash]=[(i,match_matrix[i])]

    error_groups = [(v[0][1],[_i for _i,_f in v]) for _,v in errors.items()]
    if matches is not None:
        matches = np.array([v[0][1] for _,v in matches.items()])

    error_matrix = np.array([g[0] for g in error_groups])

    error_matrix = (error_matrix-(np.mean(error_matrix,axis=0)))/np.std(error_matrix,axis=0)
    
    inherit_depth, num_cases = error_matrix[0].shape
    popsize = len(error_groups)
    rng = np.random.default_rng()

    if distribution=="normal":
        scores = rng.standard_normal(size=[n,inherit_depth,num_cases])
    elif distribution=="uniform":
        scores = rng.random(size=[n,inherit_depth,num_cases])
    elif distribution=="range":
        scores = np.array([[rng.permutation(num_cases) for _ in range(inherit_depth)] for __ in range(n)])
    else:
        raise NotImplementedError
        
    scores = scores*(std[:,:inherit_depth]/np.std(scores))+offset[:,:inherit_depth]

    weights = softmax(scores.reshape([n,-1]), axis=1)
    
    error_matrix = error_matrix.reshape([len(error_groups),-1])
    if matches is not None:
        matches = matches.reshape([len(error_groups),-1])
    res = np.matmul(error_matrix,np.transpose(weights))
    if matches is not None:
        standardize = np.matmul(matches,np.transpose(weights))
        res=res/standardize

    elite = res.argmin(axis=0)
    selected = elite

    return [rng.choice(error_groups[i][1]) for i in selected]