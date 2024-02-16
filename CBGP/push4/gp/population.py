from collections.abc import Sequence
from bisect import insort_left
from typing import Callable

import numpy as np
import pickle
from multiprocessing import Pool
from functools import partial
from itertools import takewhile

from push4.gp.individual import Individual
from push4.lang.dag import Dag
from copy import copy
import time
import ec_ecology_toolbox as eco
import json
from scipy.special import softmax 

def get_perturbation(arr, method, scale, maxsize):
    rng = np.random.default_rng()

    if scale=="percent":
        if method=="range":
            weights = np.rint(rng.uniform(low=-1,high=1,size=arr.shape)*maxsize*arr).astype(np.int32)
        elif method=="normal":
            weights = rng.normal(0,maxsize,size=arr.shape)
            weights = weights*arr
        elif method=="normalint":
            weights = np.rint(rng.normal(0,maxsize,size=arr.shape)*arr).astype(np.int32)
        else:
            raise NotImplementedError
        
    elif scale=="absolute":
        if method=="range":
            weights = rng.integers(low=-maxsize,high=maxsize,endpoint=True,size=arr.shape)
        elif method=="normal":
            weights = rng.normal(0,maxsize,size=arr.shape)
        elif method=="normalint":
            weights = np.rint(rng.normal(0,maxsize,size=arr.shape)).astype(np.int32)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return np.maximum(arr+weights,0)

def _eval_indiv(indiv: Individual, error_fn: Callable[[Dag], np.array], perturb=None):
    error = error_fn(indiv.program)
    if perturb is not None:
        err = get_perturbation(error, *perturb)
        indiv.true_error = error
        indiv.error_vector = err 
    else:
        indiv.error_vector = error

    if len(indiv.inherited_errors)==0:
        indiv.inherited_errors = np.expand_dims(indiv.error_vector,axis=0)
    else:
        indiv.inherited_errors = np.insert(indiv.inherited_errors,0,indiv.error_vector, axis=0)
    return indiv

def preselect(population):
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which individual gets selected.
    """
    population_list = list(copy(population))

    error_matrix = []
    error_vector_hashes = []
    candidates = {}

    for individual in population_list:
        error_vector_hash = hash(individual.error_vector_bytes)
        if error_vector_hash not in error_vector_hashes:
            error_vector_hashes.append(error_vector_hash)
            candidates[error_vector_hash] = [individual]
            error_matrix.append(individual.error_vector)
        else:
            candidates[error_vector_hash] += [individual]

    return np.array(error_matrix), error_vector_hashes, candidates


class Population(Sequence):
    """A sequence of Individuals kept in sorted order, with respect to their total errors."""

    __slots__ = ["unevaluated", "evaluated", "nondom_set", "parents", "logs","selector"]

    def __init__(self, individuals: list = None):
        self.unevaluated = []
        self.evaluated = []
        self.nondom_set = []
        self.logs = {
            'weighted_lexicase_runtime': 0,
            'lexicase_runtime':0,
            'evaluated_solutions':0
        }

        if individuals is not None:
            for el in individuals:
                self.add(el)

    def __len__(self):
        return len(self.evaluated) + len(self.unevaluated)

    def __getitem__(self, key: int) -> Individual:
        if key < len(self.evaluated):
            return self.evaluated[key]
        return self.unevaluated[key - len(self.evaluated)]

    def add(self, individual: Individual):
        """Add an Individual to the population."""
        if individual.total_error is None:
            self.unevaluated.append(individual)
        else:
            insort_left(self.evaluated, individual)
        return self

    def best(self):
        """Return the best n individual in the population."""
        return self.evaluated[0]

    def best_n(self, n: int):
        """Return the best n individuals in the population."""
        return self.evaluated[:n]

    def solving(self, error_fn, threshold=0, perturb=None, expected_cases=None):
        
        if perturb is not None:
            threshold = np.max([get_perturbation(np.zeros_like(self[0].error_vector), *perturb) for i in range(100)])
            print(f"THRESHOLD {threshold}")

        potential_candidates = list(takewhile(lambda x: np.all(x.inherited_errors<=threshold), self.evaluated))

        for c in potential_candidates:
            error_vector = error_fn(c.program)
            if expected_cases is not None:
                assert error_vector.numel() == expected_cases, f"error vector {error_vector} expected cases {expected_cases}"
            total_error=np.sum(c.error_vector)
            print(f"Potential solution error vector: {c.error_vector}", flush=True)
            print(f"Potential solution error sum: {np.sum(c.error_vector)}", flush=True)
            print(f"Potential solution total error: {c.total_error}", flush=True)
            self.logs["evaluated_solutions"]+=1
            if np.all(total_error<=0.000000001):
                return c
        return None

    def p_evaluate(self, error_fn, pool: Pool):
        """Evaluate all unevaluated individuals in the population in parallel."""
        func = partial(_eval_indiv, error_fn=error_fn)
        for individual in pool.imap_unordered(func, self.unevaluated):
            insort_left(self.evaluated, individual)
        self.unevaluated = []

    def plexicase(self, alpha=1, num_parents=1):
        error_matrix, error_vector_hashes, candidates = preselect(self)

        t = time.time()
        # generate dominate set
        best_err = np.min(error_matrix, axis=0)
        n_cand, n_cases = error_matrix.shape

        err_is_best = (error_matrix <= best_err).astype(float)
        n_best = np.sum(err_is_best, axis=1)
        n_nonbest = np.sum(error_matrix>best_err,axis=1)

        unchecked = np.argsort(n_best)
        dom_set = []
        total_comparisons = 0

        while len(unchecked) > 0:
            idx = unchecked[0]

            cur_n_best = n_best[idx]
            to_compare = np.arange(n_cand)[n_best <= cur_n_best]
            # remove self
            to_compare = to_compare[to_compare != idx]
            # remove already dominated cand
            to_compare = to_compare[np.logical_not(
                np.isin(to_compare, dom_set))]

            if len(to_compare) > 0:
                total_comparisons += len(to_compare)

                cur_err = error_matrix[idx]
                to_compare_err = error_matrix[to_compare]

                cur_err_is_best = err_is_best[idx]
                to_compare_err_is_best = err_is_best[to_compare]

                # A - current; B - to compare
                # tie on some best and B is better on something
                cond1 = (np.sum(cur_err_is_best * to_compare_err_is_best,
                                axis=1) *
                         np.sum((to_compare_err < cur_err), axis=1)
                         ) > 0

                # B is better on something with best error
                cond2 = np.sum(to_compare_err_is_best *
                               (1-cur_err_is_best), axis=1) > 0

                dom_set += list(to_compare[np.logical_not(np.logical_or(cond1, cond2))])

            unchecked = unchecked[unchecked != idx]  # remove self
            unchecked = unchecked[np.logical_not(np.isin(unchecked, dom_set))]

        nondom_set = np.arange(n_cand)
        nondom_set = nondom_set[np.logical_not(np.isin(nondom_set, dom_set))]
        n_best_nondom = n_best[nondom_set]
        n_nonbest_nondom=n_nonbest[nondom_set]

        best_each_case = np.array(
            [error_matrix[nondom_set, i] == best_err[i] for i in range(n_cases)]).astype(float)  # (n_cases, nondom set)
        p_each_case = best_each_case * n_best_nondom #/(n_nonbest_nondom+0.1) #n_best_nondom
        # p_each_case=softmax(p_each_case, axis=1)
        p_each_case = p_each_case / np.sum(p_each_case, axis=1, keepdims=True)
        p = np.sum(p_each_case, axis=0)

        # err = error_matrix[nondom_set]
        # p0 = np.expand_dims(err,1)
        # better = np.sum(err<p0, axis=-1)
        # worse = np.sum(err>p0,axis=-1)

        # pworse = np.where(worse==0, 1, worse)
        # pbetter = np.where(worse==0, 1, better)

        # p_selection  = pbetter/pworse
        # p_selection/=np.sum(p_selection,axis=-1)
        # p_selection = np.mean(p_selection,axis=0)
        # p=p_selection

        # normalize
        p = p/np.sum(p)

        # manipulate
        p = np.power(p, alpha)
        p = p/np.sum(p)

        candidates = [candidates[error_vector_hashes[i]] for i in nondom_set]
        self.nondom_set = (candidates, p)

        self.parents = np.random.choice(
            np.arange(len(candidates)), p=p, replace=True, size=num_parents
        )
        self.parents = [np.random.choice(candidates[i]) for i in self.parents]

        delta = time.time()-t
        # print(f"Lexiprob runtime {delta}")
        self.logs['lexiprob_runtime'] = delta
        self.logs['lexiprob_comparisons'] = total_comparisons
        return self.parents


    def evaluate(self, error_fn: Callable[[Dag], np.array], perturb=None):
        """Evaluate all unevaluated individuals in the population."""

        for i, individual in enumerate(self.unevaluated):
            individual = _eval_indiv(individual, error_fn, perturb=perturb)
            insort_left(self.evaluated, individual)
        self.unevaluated = []

    def all_error_vectors(self, idx=None):
        """2D array containing all Individuals' error vectors."""
        if idx is None:
            vec = np.vstack([i.error_vector for i in self.evaluated])  
        else:
            vec = np.vstack([np.array(i.inherited_errors[idx]) for i in self.evaluated])
        return vec

    def all_total_errors(self):
        """1D array containing all Individuals' total errors."""
        return np.array([i.total_error for i in self.evaluated])

    def median_error(self):
        """Median total error in the population."""
        try:
            med = np.median(self.all_total_errors())
        except Exception as e:
            med = -1
            print("MEDIAN ERROR OCCURRED")
            print(e)
            print(f"TOTAL ERRORS {self.all_total_errors()}")
            print(f"DTYPE TOTAL {self.all_total_errors().dtype} TOTAL[0] {self.all_total_errors()[0].dtype} EVALUATED[0].total_error {self.evaluated[0].total_error}")

        return med

    def error_diversity(self):
        """Proportion of unique error vectors."""
        try:
            unique = np.unique(self.all_error_vectors(), axis=0)
        except Exception as e:
            unique=[]
            print("UNIQUE ERROR OCCURRED")
            print(e)
            print(f"TOTAL ERRORS {self.all_error_vectors()}")
            print(f"DTYPE TOTAL {self.all_error_vectors().dtype} TOTAL[0] {self.all_error_vectors()[0].dtype} EVALUATED[0] {self.evaluated[0].error_vector.dtype} EVALUATED[0].error {self.evaluated[0].error_vector}")

        return len(unique) / float(len(self))

    def genome_diversity(self):
        """Proportion of unique genomes."""
        unq = set([pickle.dumps(i.genome) for i in self])
        return len(unq) / float(len(self))

    def program_diversity(self):
        """Proportion of unique programs."""
        unq = set([pickle.dumps(i.get_program().code) for i in self])
        return len(unq) / float(len(self))

    def find_index(self, ind, candidates):
        for i,c in enumerate(candidates):
            for ind2 in c:
                if ind==ind2:
                    return i,len(c)
        return None,None

    def find_prob(self, ind, error_groups, values, counts):
        for i,(_,g) in enumerate(error_groups):
            for ind2 in g:
                if ind2==ind:
                    idx = np.where(values==i)[0]
                    if len(idx)==0:
                        return 0
                    else:
                        return float(counts[idx[0]]/50000/len(g))

        print("INDIVIDUAL NOT FOUND, RETURNING 0")
        return 0

    def json(self, wlexicase_selector, lexicase_selector, sample_size=50000):
        if len(self.nondom_set)==0:
            self.plexicase()
        candidates, p = self.nondom_set
        l = np.array([-i.error_vector for i in self.evaluated])
        for ind in self.evaluated:
            i,l = self.find_index(ind,candidates)
            if i is not None:
                ind.pprob=float(p[i]/l)
            else:
                ind.pprob=0
        
        t = time.time()
        error_groups, values, counts = lexicase_selector.error_frequencies(self,sample_size)
        # lexicase_fitnesses = eco.LexicaseFitness([-i.error_vector for i in self.evaluated])
        for ind in self.evaluated:
            lprob = self.find_prob(ind,error_groups, values, counts)
            ind.lprob = lprob
        print(f"LEXICASE TEST TIME: {time.time()-t}", flush=True)
        

        t = time.time()
        error_groups, values, counts = wlexicase_selector.error_frequencies(self,sample_size)
        for ind in self.evaluated:
            wprob = self.find_prob(ind,error_groups, values, counts)
            # print(f"wprob: {wprob}, lprob: {ind.lprob}, pprob: {ind.pprob}")
            ind.wprob = wprob
        print(f"WLEXICASE TEST TIME: {time.time()-t}", flush=True)


        return [ind.json() for ind in self.evaluated]
