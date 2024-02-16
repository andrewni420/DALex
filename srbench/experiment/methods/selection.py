"""The :mod:`selection` module defines classes to select Individuals from Populations."""
from abc import ABC, abstractmethod

import numpy as np
from operator import attrgetter
from typing import Sequence, Union
import time
from scipy.special import softmax

def median_absolute_deviation(x: np.array) -> float:
    """Return the MAD.

    Parameters
    ----------
    x : array-like, shape = (n,)

    """
    return np.median(np.abs(x - np.median(x))).item()


class Selector(ABC):
    """Base class for all selection algorithms."""
    def __init__(self):
        self.rng = np.random.default_rng()

    @abstractmethod
    def select_one(self, population):
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        pass

    def select(self, population, n: int = 1) -> Sequence:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population : Population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        selected = []
        for i in range(n):
            selected.append(self.select_one(population))
        return selected


class FitnessProportionate(Selector):
    """Fitness proportionate selection, also known as roulette wheel selection.

    See: https://en.wikipedia.org/wiki/Fitness_proportionate_selection
    """

    def select_one(self, population):
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        return self.select(population)[0]

    def select(self, population, n: int = 1) -> Sequence:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        population_total_errors = np.array([i.total_error for i in population])
        sum_of_total_errors = np.sum(population_total_errors)
        probabilities = 1.0 - (population_total_errors / sum_of_total_errors)
        selected_ndxs = np.searchsorted(np.cumsum(probabilities), self.rng.choice(n))
        return [population[ndx] for ndx in selected_ndxs]


class Tournament(Selector):
    """Tournament selection.

    See: https://en.wikipedia.org/wiki/Tournament_selection

    Parameters
    ----------
    tournament_size : int, optional
        Number of individuals selected uniformly randomly to participate in
        the tournament. Default is 7.

    Attributes
    ----------
    tournament_size : int, optional
        Number of individuals selected uniformly randomly to participate in
        the tournament. Default is 7.

    """

    def __init__(self, tournament_size: int = 7):
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng()

    def select_one(self, population):
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        tournament = self.rng.choice(population, self.tournament_size, replace=False)
        return min(tournament, key=attrgetter('total_error'))


class CaseStream:

    def __init__(self, n_cases: int):
        self.cases = list(range(n_cases))

    def __iter__(self):
        rng = np.random.default_rng()
        rng.shuffle(self.cases)
        for case in self.cases:
            yield case


def one_individual_per_error_vector(population) -> Sequence:
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which indiviudal gets selected.
    """
    rng = np.random.default_rng()
    population_list = list(population)
    rng.shuffle(population_list)
    preselected = []
    error_vector_hashes = set()
    for individual in population_list:
        error_vector_hash = hash(individual.inherit_error_bytes)
        if error_vector_hash not in error_vector_hashes:
            preselected.append(individual)
            error_vector_hashes.add(error_vector_hash)
    return preselected

def one_individual_per_actual_error_vector(population) -> Sequence:
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which indiviudal gets selected.
    """
    rng = np.random.default_rng()
    population_list = list(population)
    rng.shuffle(population_list)
    preselected = []
    error_vector_hashes = set()
    for individual in population_list:
        error_vector_hash = hash(individual.error_vector_bytes)
        if error_vector_hash not in error_vector_hashes:
            preselected.append(individual)
            error_vector_hashes.add(error_vector_hash)
    return preselected

def distinct_errors(population):
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which indiviudal gets selected.
    """
    errors = dict()
    population_list = list(population)
    for individual in population_list:
        error_vector_hash = hash(individual.inherit_error_bytes)
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append(individual)
        else:
            errors[error_vector_hash]=[individual]

    return [[v[0].inherited_errors, v] for _,v in errors.items()]


def lexicase(fitnesses, n=1, alpha=1, epsilon=False):
    errors = dict()
    for i,f in enumerate(fitnesses):
        error_vector_hash = hash(f.tobytes())
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append((i,f))
        else:
            errors[error_vector_hash]=[(i,f)]

    error_groups = [(v[0][1],[_i for _i,_f in v]) for _,v in errors.items()]

    error_matrix = np.array([g[0] for g in error_groups])
    # print(np.round(error_matrix[:10,0,:10],decimals=2))
    
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

    select_counts = []

    for _ in range(n):
        count = 0
        candidates = range(popsize)
        for i in range(inherit_depth):
            if len(candidates) <= 1:
                break
            ordering = rng.permutation(num_cases)
            for case in ordering:
                if len(candidates) <= 1:
                    break
                count +=1
                errors_this_case = [error_matrix[ind][i][case] for ind in candidates]
                best_val_for_case = min(errors_this_case)+ep[i][case]
                candidates = [ind for ind in candidates if error_matrix[ind][i][case] <= best_val_for_case]
        selected.append(rng.choice(candidates))
        select_counts.append(count)

    # print(np.mean(counts))
    values, counts = np.unique(selected,return_counts=True)

    p=counts/np.sum(counts)
    p=p**alpha 
    p=p/np.sum(p)

    selected = rng.choice(values,p=p,replace=True, size=n)

    return [rng.choice(error_groups[i][1]) for i in selected], select_counts


def wlexicase(fitnesses, n=1, alpha=1, std=np.array(1).reshape([1,1,1]), offset=np.array(0).reshape([1,1,1]), distribution="normal", epsilon=False):
    std = np.array(std).astype(np.float32)
    offset=np.array(offset).astype(np.float32)
    std = std if std.ndim==3 else np.reshape(std,[1,std.shape[0] if std.ndim==1 else 1,1])
    offset = offset if offset.ndim==3 else np.reshape(offset,[1,offset.shape[0]if offset.ndim==1 else 1,1])
    errors = dict()
    for i,f in enumerate(fitnesses):
        error_vector_hash = hash(f.tobytes())
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append((i,f))
        else:
            errors[error_vector_hash]=[(i,f)]

    error_groups = [(v[0][1],[_i for _i,_f in v]) for _,v in errors.items()]

    error_matrix = np.array([g[0] for g in error_groups])

    error_matrix = (error_matrix-(np.mean(error_matrix,axis=0)))/np.std(error_matrix,axis=0)
    
    inherit_depth, num_cases = error_matrix[0].shape
    popsize = len(error_groups)
    rng = np.random.default_rng()

    t = time.time()
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
    elite = np.matmul(error_matrix,np.transpose(weights)).argmin(axis=0)
    values, counts = np.unique(elite,return_counts=True)

    p=counts/np.sum(counts)
    p=p**alpha 
    p=p/np.sum(p)

    selected = rng.choice(values,p=p,replace=True, size=n)

    return [rng.choice(error_groups[i][1]) for i in selected]

class Lexicase(Selector):
    """Lexicase Selection.

    All training cases are considered iteratively in a random order. For each
    training cases, the population is filtered to only contain the Individuals
    which have an error value within epsilon of the best error value on that case.
    This filtering is repeated until the population is down to a single Individual
    or all cases have been used. After the filtering iterations, a random
    Individual from the remaining set is returned as the selected Individual.

    See: https://ieeexplore.ieee.org/document/6920034
    """

    def __init__(self, epsilon: Union[bool, float, np.ndarray] = False, alpha=1.):
        self.epsilon = epsilon
        self.lexicase_inherit_depth = []
        self.lexicase_case_depth = []
        self.rng = np.random.default_rng()
        self.alpha = alpha

    @staticmethod
    def _epsilon_from_mad(error_matrix: np.ndarray):
        return np.median(np.abs(error_matrix - np.median(error_matrix,axis=0)),axis=0)

    def _filter_with_stream(self, error_groups, cases: CaseStream, i:int=0, ep=False):
        idx=0
        for case in cases:
            if len(error_groups) <= 1:
                return error_groups,idx

            errors_this_case = [e[i][case] for e,_ in error_groups]
            best_val_for_case = min(errors_this_case)

            max_error = best_val_for_case
            if isinstance(ep, np.ndarray):
                max_error += ep[case]
            elif isinstance(ep, (float, int, np.int64, np.float64)):
                max_error += ep

            error_groups = [
                [e,_] for e,_ in error_groups if e[i][case] <= max_error]
        
            idx+=1
        return error_groups,idx

    def lexicase_by_error(self, error_matrix, best_errors):
        inherit_depth, num_cases = error_matrix[0].shape
        popsize = len(error_matrix)
        rng = np.random.default_rng()
        candidates = range(popsize)
        non_global=0
        total=0
        for i in range(inherit_depth):
            if len(candidates) <= 1:
                break
            ordering = rng.permutation(num_cases)
            for case in ordering:
                if len(candidates) <= 1:
                    break
                errors_this_case = [error_matrix[ind][i][case] for ind in candidates]
                best_val_for_case = min(errors_this_case)
                candidates = [ind for ind in candidates if error_matrix[ind][i][case] <= best_val_for_case]
                total+=1
                if best_val_for_case!=best_errors[i][case]:
                    non_global+=1
        self.percent_non_global.append(non_global/total)
        self.total.append(total)
        return rng.choice(candidates)

    
    def error_frequencies(self, population,n=1):
        self.percent_non_global=[]
        self.total=[]
        error_groups = distinct_errors(population)
        inherit_depth,num_cases = population[0].inherited_errors.shape

        error_matrix = [e for e,_ in error_groups]
        best_errors=np.min(error_matrix,axis=0)

        elite = [self.lexicase_by_error(error_matrix, best_errors) for _ in range(n)]
        print(f"percent non global: {np.mean(self.percent_non_global)}")
        print(f"total: {np.mean(self.total)}")

        values, counts = np.unique(elite,return_counts=True)
        return error_groups,values,counts


    def select_one(self, error_groups, ep):
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        num_inherited,num_cases = error_groups[0][0].shape
        candidates = error_groups

        for i in range(num_inherited):
            cases = CaseStream(num_cases)
            candidates,j = self._filter_with_stream(candidates, cases,i=i, ep=ep[i])
            if len(candidates)<=1:
                self.lexicase_inherit_depth.append(i)
                self.lexicase_case_depth.append(i*num_cases+j)
                break

        r = candidates[self.rng.choice(len(candidates))][1]
        return r[self.rng.choice(len(r))]

    def select(self, population, n: int = 1) -> Sequence:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population : Population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        num_cases = len(population[0].inherited_errors[0])
        num_inherited = len(population[0].inherited_errors)
        error_groups = distinct_errors(population)

        ep = self.epsilon
        if isinstance(ep, bool) and ep:
            ep = self._epsilon_from_mad(np.array([p.inherited_errors for p in population]))
        else:
            ep = [ep]*num_inherited

        t = time.time()
        selected = []
        selected = [self.select_one(error_groups, ep) for i in range(n)]
        # candidates = one_individual_per_error_vector(population)
        # cand2 = one_individual_per_actual_error_vector(population)
        
        # print(f"Distinct error count {len(candidates)}")
        # print(f"Distinct actual error count {len(cand2)}")
        # print(f"Inherit error depth {candidates[0].inherited_errors.shape}")
        # print(f"Distinct parents {np.unique([int((c.parent_id or [-1])[0]) for c in candidates])}")
        # print(f"Average lexicase case depth: {np.mean(self.lexicase_case_depth)}")
        # print(f"Average lexicase inherit depth: {np.mean(self.lexicase_inherit_depth)}")

        self.lexicase_case_depth=[]
        self.lexicase_inherit_depth=[]

        return selected

class PLexicase(Selector):
    def __init__(self, alpha=1):
        self.alpha=1
    def select(self, population, n: int = 1) -> Sequence:
        return population.plexicase(alpha=self.alpha, num_parents=n)
    def select_one(self, population):
        return self.select(population)


class WeightedLexicase(Selector):
    """Lexicase Selection via weighted generalized average.

    All training cases are considered iteratively in a random order. For each
    training cases, the population is filtered to only contain the Individuals
    which have an error value within epsilon of the best error value on that case.
    This filtering is repeated until the population is down to a single Individual
    or all cases have been used. After the filtering iterations, a random
    Individual from the remaining set is returned as the selected Individual.

    See: https://ieeexplore.ieee.org/document/6920034
    """

    def __init__(self, std=np.array(20).reshape([1,1,1]), offset=np.array(0).reshape([1,1,1]), distribution:str ="normal"):
        self.std = std
        self.offset = offset
        self.distribution = distribution
        assert self.distribution in ["normal", "uniform", "range"]

    def error_frequencies(self, population,n=1):
        error_groups = distinct_errors(population)
        inherit_depth,num_cases = population[0].inherited_errors.shape

        rng = np.random.default_rng()

        t = time.time()
        if self.distribution=="normal":
            scores = rng.standard_normal(size=[n,inherit_depth,num_cases])
        elif self.distribution=="uniform":
            scores = rng.random(size=[n,inherit_depth,num_cases])
        elif self.distribution=="range":
            scores = np.array([[rng.permutation(num_cases) for _ in range(inherit_depth)] for __ in range(n)])
        else:
            raise NotImplementedError


        scores = scores*(self.std[:,:inherit_depth]/np.std(scores,axis=0))+self.offset[:,:inherit_depth]
        

        weights = softmax(scores.reshape([n,-1]).astype(np.float128), axis=1)
        

        error_matrix = np.stack([e for e,_ in error_groups],dtype=np.float128).reshape([len(error_groups),-1])
        elite = np.matmul(error_matrix,np.transpose(weights)).argmin(axis=0)
        values, counts = np.unique(elite,return_counts=True)
        return error_groups,values,counts


    def select(self, population, n: int = 1) -> Sequence:
        t = time.time()
        error_groups = distinct_errors(population)
        inherit_depth,num_cases = population[0].inherited_errors.shape

        rng = np.random.default_rng()

       
        if self.distribution=="normal":
            scores = rng.standard_normal(size=[n,inherit_depth,num_cases])
        elif self.distribution=="uniform":
            scores = rng.random(size=[n,inherit_depth,num_cases])
        elif self.distribution=="range":
            scores = np.array([[rng.permutation(num_cases) for _ in range(inherit_depth)] for __ in range(n)])
        else:
            raise NotImplementedError


        scores = scores*(self.std[:,:inherit_depth]/np.std(scores))+self.offset[:,:inherit_depth]

        weights = softmax(scores.reshape([n,-1]), axis=1)

        error_matrix = np.stack([e for e,_ in error_groups]).reshape([len(error_groups),-1])
        elite = np.matmul(error_matrix,np.transpose(weights)).argmin(axis=0)

        parents = [rng.choice(error_groups[i][1]) for i in elite]

        delta = time.time() - t
        print(f"wlexicase runtime: {delta}", flush=True)
        population.logs['weighted_lexicase_runtime'] += delta

        return parents

    
    # def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
    #     error_groups = distinct_errors(population)
    #     num_cases = len(population[0].error_vector)
    #     rng = np.random.default_rng()

    #     t = time.time()
    #     if self.distribution=="normal":
    #         scores = rng.standard_normal(size=[n,num_cases])
    #     elif self.distribution=="uniform":
    #         scores = rng.random(size=[n,num_cases])
    #     elif self.distribution=="range":
    #         scores = np.array([rng.permutation(num_cases) for _ in range(n)])
    #     else:
    #         raise NotImplementedError

    #     scores = scores*(self.std/np.std(scores))
    #     weights = softmax(scores.astype(np.float128), axis=1)

    #     error_matrix = np.stack([e for e,_ in error_groups],dtype=np.float128)
    #     elite = np.matmul(error_matrix,np.transpose(weights)).argmin(axis=0)

    #     parents = [rng.choice(error_groups[i][1]) for i in elite]

    #     delta = time.time() - t
    #     population.logs['weighted_lexicase_runtime'] += delta
    #     print(f"wlexicase runtime: {delta}")

    #     return parents

    def select_one(self, population):
        return self.select(population, 1)[0]

#[pop_size, cases] x [cases, n]-> #[n, pop_size]
class Elite(Selector):
    """Returns the best N individuals by total error."""

    def select_one(self, population):
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        return population.best()

    def select(self, population, n: int = 1) -> Sequence:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        return population.best_n(n)
