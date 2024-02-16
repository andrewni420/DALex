"""The :mod:`selection` module defines classes to select Individuals from Populations."""
from abc import ABC, abstractmethod
from copy import copy
from typing import Sequence, Union
from operator import attrgetter

import numpy as np
from numpy.random import random, choice, shuffle

from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.tap import tap
from pyshgp.utils import instantiate_using
from scipy.special import softmax


def distinct_errors(population: Population):
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which indiviudal gets selected.
    """
    errors = dict()
    population_list = list(copy(population))
    for individual in population_list:
        error_vector_hash = hash(individual.inherit_error_bytes)
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append(individual)
        else:
            errors[error_vector_hash]=[individual]

    return [[h, v[0].inherited_errors, v] for h,v in errors.items()]


class Selector(ABC):
    def __init__(self):
        self.rng = np.random.default_rng()

    """Base class for all selection algorithms."""

    @abstractmethod
    def select_one(self, population: Population) -> Individual:
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

    @abstractmethod
    @tap
    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
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
        pass



class SimpleMultiSelectorMixin:
    """A mixin for ``Selector`` classes where selecting many individuals is done by repeated calls to `select_one`."""

    @tap
    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
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

    def select_one(self, population: Population) -> Individual:
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

    @tap
    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
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
        super().select(population, n)
        population_total_errors = np.array([i.total_error for i in population])
        sum_of_total_errors = np.sum(population_total_errors)
        probabilities = 1.0 - (population_total_errors / sum_of_total_errors)
        selected_ndxs = np.searchsorted(np.cumsum(probabilities), random(n))
        return [population[ndx] for ndx in selected_ndxs]

class Tournament(SimpleMultiSelectorMixin, Selector):
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

    def select_one(self, population: Population) -> Individual:
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
        tournament = choice(population, self.tournament_size, replace=False)
        return min(tournament, key=attrgetter('total_error'))


def median_absolute_deviation(x: np.ndarray) -> np.float64:
    """Return the MAD.

    Parameters
    ----------
    x : array-like, shape = (n,)

    Returns
    -------
    mad : float

    """
    return np.median(np.abs(x - np.median(x))).item()

class CaseStream:
    """A generator of indices yielded in a random order."""

    # @todo generalize to RandomIndexStream

    def __init__(self, n_cases: int):
        self.cases = list(range(n_cases))

    def __iter__(self):
        shuffle(self.cases)
        for case in self.cases:
            yield case


def one_individual_per_error_vector(population: Population) -> Sequence[Individual]:
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which individual gets selected.
    """
    population_list = list(copy(population))
    shuffle(population_list)
    preselected = []
    error_vector_hashes = set()
    for individual in population_list:
        error_vector_hash = hash(individual.error_vector_bytes)
        if error_vector_hash not in error_vector_hashes:
            preselected.append(individual)
            error_vector_hashes.add(error_vector_hash)
    return preselected

        
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

    def __init__(self, alpha:int=1, epsilon: Union[bool, float, np.ndarray] = False):
        super().__init__(alpha=alpha)
        self.epsilon = epsilon

    @staticmethod
    def _epsilon_from_mad(error_matrix: np.ndarray):
        return np.apply_along_axis(median_absolute_deviation, 0, error_matrix)

    def lexicase_by_error(self, error_matrix):
        inherit_depth, num_cases = error_matrix[0].shape
        rng = np.random.default_rng()
        popsize = len(error_matrix)
        candidates = range(popsize)
        for i in range(inherit_depth):
            if len(candidates) <= 1:
                break
            ordering = self.rng.permutation(num_cases)
            for case in ordering:
                if len(candidates) <= 1:
                    break
                errors_this_case = [error_matrix[ind][i][case] for ind in candidates]
                best_val_for_case = min(errors_this_case)
                candidates = [ind for ind in candidates if error_matrix[ind][i][case] <= best_val_for_case]
        return self.rng.choice(candidates)

    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        error_matrix = [i.inherited_errors for i in population]
        selected = lexicase(error_matrix,n=n,epsilon=self.epsilon)
        return [population[i] for i in selected]
    
    def select_one(self, population: Population) -> Individual:
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
        error_matrix = [i.inherited_errors for i in population]
        return population[self.lexicase_by_error(error_matrix)]

    def error_frequencies(self, population,n=1):
        error_groups = distinct_errors(population)
        inherit_depth,num_cases = population[0].inherited_errors.shape

        error_matrix = [e for _,e,_ in error_groups]

        elite = lexicase(error_matrix,n=n,epsilon=self.epsilon)

        values, counts = np.unique(elite,return_counts=True)
        return [g[1:] for g in error_groups],values,counts

    def sample(self, error_matrix, n:int=1):
        return [self.lexicase_by_error(error_matrix) for _ in range(n)]

class Elite(Selector):
    """Returns the best N individuals by total error."""

    def select_one(self, population: Population) -> Individual:
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

    @tap
    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
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
        super().select(population, n)
        return population.best_n(n)

def get_selector(name: str, **kwargs) -> Selector:
    """Get the selector class with the given name."""
    name_to_cls = {
        "roulette": FitnessProportionate,
        "tournament": Tournament,
        "lexicase": Lexicase,
        "epsilon-lexicase": Lexicase(epsilon=True),
        "elite": Elite,
        "dalex":DALex,
        "plexicase":PLexicase,
    }
    selector = name_to_cls.get(name.lower(), None)
    if selector is None:
        raise ValueError("No selector '{nm}'. Supported names: {lst}.".format(
            nm=name,
            lst=list(name_to_cls.keys())
        ))
    if isinstance(selector, type):
        selector = instantiate_using(selector, kwargs)
    return selector


class PLexicase(Selector):
    def __init__(self, alpha=1):
        self.alpha=1
    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        return population.plexicase(alpha=self.alpha, num_parents=n)
    def select_one(self, population):
        return self.select(population)


class DALex(Selector):
    """Lexicase Selection via weighted generalized average.

    All training cases are considered iteratively in a random order. For each
    training cases, the population is filtered to only contain the Individuals
    which have an error value within epsilon of the best error value on that case.
    This filtering is repeated until the population is down to a single Individual
    or all cases have been used. After the filtering iterations, a random
    Individual from the remaining set is returned as the selected Individual.

    See: https://ieeexplore.ieee.org/document/6920034
    """

    def __init__(self, std:float = 20, distribution:str = "normal"):
        super().__init__()
        self.std = std
        self.distribution = distribution
        assert self.distribution in ["normal", "uniform", "range"]

    def select_one(self, population):
        return super().select(population,1)[0]
    
    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        error_matrix = [i.inherited_errors for i in population]
        selected = dalex(error_matrix,n=n,std=self.std,distribution=self.distribution)
        return [population[i] for i in selected]
    
    def error_frequencies(self, population,n=1):
        error_groups = distinct_errors(population)
        inherit_depth,num_cases = population[0].inherited_errors.shape

        error_matrix = [e for _,e,_ in error_groups]

        elite = dalex(error_matrix,n=n,std=self.std,distribution=self.distribution)

        values, counts = np.unique(elite,return_counts=True)
        return [g[1:] for g in error_groups],values,counts
        
    def sample(self, error_matrix, n):
        inherit_depth,num_cases = error_matrix[0].shape
        if self.distribution=="normal":
            scores = self.rng.standard_normal(size=[n,inherit_depth,num_cases])
        elif self.distribution=="uniform":
            scores = self.rng.random(size=[n,inherit_depth,num_cases])
        elif self.distribution=="range":
            scores = np.array([[self.rng.permutation(num_cases) for _ in range(inherit_depth)] for __ in range(n)])
        else:
            raise NotImplementedError

        scores = scores*(self.std/np.std(scores))

        weights = softmax(scores.reshape([n,-1]).astype(np.float128), axis=1)

        return np.matmul(np.array(error_matrix).reshape([len(error_matrix),-1]),np.transpose(weights)).argmin(axis=0)

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


def dalex(fitnesses:np.ndarray, n:int = 1, std:float = 50., distribution:str = "normal"):
    '''Takes fitnesses as an m x n numpy array for a population of m individuals evaluated on 
       n test cases and returns the indices of the selected error vectors.
       Optional parameters
          n: The number of selection events to compute in one batch. Returns the n selected indices
          std: The particularity pressure parameter. Controls the standard deviation of the sampled importance scores
          distribution: The distribution from which to draw the importance scores. Currently supported options are
                normal: The default option, which samples from a normal distribution
                uniform: Samples from a uniform distribution
                range: Shuffles an evenly spaced range of numbers to get the importance scores
        Returns a length n integer array of the n indices selected by DALex.'''

    #sort error vectors into equivalence classes of identical error vectors
    errors = dict()
    for i,f in enumerate(fitnesses):
        error_vector_hash = hash(f.tobytes())
        if error_vector_hash in errors.keys():
            errors[error_vector_hash].append((i,f))
        else:
            errors[error_vector_hash]=[(i,f)]
    #list of tuples [error vector, indices of identical error vectors in fitnesses array]
    error_groups = [(v[0][1],[_i for _i,_f in v]) for _,v in errors.items()]

    #Reconstruct the fitness matrix without duplicate error vectors
    error_matrix = np.array([g[0] for g in error_groups]).astype(np.float128)

    error_vector_shape =  error_matrix[0].shape
    popsize = len(error_groups)
    rng = np.random.default_rng()

    #Sample importance scores
    if distribution=="normal":
        scores = rng.standard_normal(size=[n,*error_vector_shape])
    elif distribution=="uniform":
        scores = rng.random(size=[n,*error_vector_shape])
    elif distribution=="range":
        scores = np.array([rng.permutation(num_cases) for __ in range(n)])
    else:
        raise NotImplementedError
        
    #Coerce importance scores to the specified particularity pressure
    scores = scores*(std/np.std(scores))+offset
    scores = scores.astype(np.float128)

    #To obtain the test case weights, softmax the importance scores
    weights = softmax(scores.reshape([n,-1]), axis=1)

    #Matrix multiply errors with test case weights and take the argmin to get the index of the selected error vector equivalence class
    error_matrix = error_matrix.reshape([len(error_groups),-1])
    selected = np.matmul(error_matrix,np.transpose(weights)).argmin(axis=0)

    #From each error vector equivalence class, choose a random index.
    return [rng.choice(error_groups[i][1]) for i in selected]