from abc import abstractmethod, ABC
from typing import Callable, Tuple

import numpy as np
import sys
import time

from push4.gp.individual import Individual
from push4.gp.population import Population, _eval_indiv
from push4.gp.selection import Selector, WeightedLexicase, Lexicase, lexicase, wlexicase
from push4.gp.spawn import Spawner
from push4.gp.variation import VariationOperator, umad
from push4.lang.dag import Dag
from push4.utils import escape

from .maxbandit import MaxBanditAO

global_id=0


def _spawn_individual(spawner, genome_size, output_type: type, *args):
    global global_id
    ind_id = str(global_id)
    global_id+=1
    genome = spawner.spawn_genome(*genome_size)
    ind = Individual(genome, output_type,ind_id=ind_id)
    # print(ind.program)
    return ind


class Evolver(ABC):

    def __init__(self,
                 downsampler: Callable,
                 error_function: Callable[[Dag], np.array],
                 spawner: Spawner,
                 selector: Selector,
                 variation: VariationOperator,
                 population_size: int,
                 max_generations: int,
                 initial_genome_size: Tuple[int, int],
                 downsample_rate: float = 1.,
                 alpha: float = 1.,
                 inherit_depth: int=0,
                 fitnesses=None,
                 perturb=None,
                 sample_size=0):
        self.downsampler = downsampler
        self.error_function = error_function
        self.spawner = spawner
        self.selector = selector
        self.variation = variation
        self.population_size = population_size
        self.max_generations = int(max_generations/downsample_rate)
        self.initial_genome_size = initial_genome_size
        self.population = None
        self.generation = 0
        self.best_seen = None
        self._is_solved=False
        self.perturb=perturb
        self.downsample_rate = downsample_rate
        self.alpha = alpha  # lexiprob
        self.inherit_depth = inherit_depth
        self.logs = {
            'lexiprob_runtime': [],
            'lexiprob_comparisons': [],
            'lexicase_runtime': [],
            'weighted_lexicase_runtime':[],
            'unique_overlap': [],
            'total_overlap': [],
            'eval_runtime': [],
            'produce_runtime': [],
            'total_runtime': [],
            'downsample_rate': downsample_rate,
            'alpha': alpha,
            'train_err': [],
            'selection_runtime':[],
            'evaluated_solutions':[]
        }
        self.fitnesses = fitnesses
        self.sample_size=sample_size


        # TODO: Add ParallelContext

    def init_population(self, output_type: type):
        """Initialize the population."""
        self.population = Population()
        for i in range(self.population_size):
            self.population.add(_spawn_individual(
                self.spawner, self.initial_genome_size, output_type))

    @abstractmethod
    def step(self, output_type: type):
        """Perform one generation (step) of evolution. Return if should continue.
        The step method should assume an evaluated Population, and must only
        perform parent selection and variation (producing children). The step
        method should modify the search algorithms population in-place, or
        assign a new Population to the population attribute.
        """
        pass

    def _full_step(self, output_type) -> bool:
        t = time.time()
        self.downsampler(self.downsample_rate)
        self.generation += 1
        self.population.evaluate(
            self.error_function, perturb=self.perturb)
        self.logs['eval_runtime'].append(time.time()-t)

        if self.fitnesses is not None:
            if isinstance(self.selector,WeightedLexicase):
                wlexicase_selector = self.selector
            else:
                wlexicase_selector = WeightedLexicase()
            if isinstance(self.selector,Lexicase):
                lexicase_selector = self.selector 
            else:
                lexicase_selector = Lexicase()
            self.fitnesses.append(self.population.json(wlexicase_selector, lexicase_selector, sample_size=self.sample_size))


        total_errors = [np.sum(i.error_vector) for i in self.population] if self.perturb is None else [np.sum(i.true_error) for i in self.population]
        best_idx = min(range(len(total_errors)), key=lambda x:total_errors[x])

        best_this_gen = self.population[best_idx]
        if self.best_seen is None or total_errors[best_idx] < (self.best_seen.total_error if self.perturb is None else np.sum(self.best_seen.true_error)):
            self.best_seen = best_this_gen

        best_is_valid = self.best_seen.program is not None
        print("{gn}\t\t{me}\t\t{be}\t\t{dv}\t\t{best_err}\t\t{best_code}".format(
            gn=round(self.generation, 3),
            me=round(np.median(total_errors), 3),
            be=round(total_errors[best_idx], 3),
            dv=round(self.population.error_diversity(), 3),
            best_err=(self.best_seen.total_error if self.perturb is None else np.sum(self.best_seen.true_error)),
            best_code=escape(self.best_seen.program.root.to_code()
                             ) if best_is_valid else "NA", flush=True
        ))
        # self.best_seen.program.pprint()

        self.logs['train_err'].append(float(total_errors[best_idx]))

        solving_individual = self.population.solving(lambda x:self.error_function(x, downsampled=False))
        if solving_individual is not None:
            self._is_solved=True 
            self.best_seen = solving_individual 

        if self._is_solved:
            return False

        self.step(output_type)
        self.logs['total_runtime'].append(time.time()-t)

        return True


    def run(self, output_type: type) -> Individual:
        """Run the algorithm until termination."""
        self.init_population(output_type)

        print("Gen\t\tMedian\t\tBest\t\tDiv\t\tRun Best\t\tCode", flush=True)
        while self._full_step(output_type):
            eval_gens = sum(self.logs["evaluated_solutions"]) / self.population_size / self.downsample_rate
            if self.generation + eval_gens >= self.max_generations:
                break

        if self._is_solved:
            print("Solution found.", flush=True)
        else:
            print("No solution found.", flush=True)

        return self.best_seen


class GeneticAlgorithm(Evolver):
    """Genetic algorithm to synthesize Push programs.
    An initial Population of random Individuals is created. Each generation
    begins by evaluating all Individuals in the population. Then the current
    Population is replaced with children produced by selecting parents from
    the Population and applying VariationOperators to them.
    """

    def _make_child(self, output_type: type, parents) -> Individual:
        global global_id
        parent_genomes = [p.genome for p in parents]
        ids = [p.ind_id for p in parents]
        child_genome = self.variation.produce(parent_genomes, self.spawner)
        ind_id=str(global_id)
        global_id+=1
        if len(parents)>1:
            return Individual(child_genome, output_type,ind_id=ind_id, parent_id=ids)
        else: 
            inherited_errors = parents[0].inherited_errors[:self.inherit_depth]
            return Individual(child_genome, output_type, inherited_errors=inherited_errors, ind_id=ind_id, parent_id = ids)

    def step(self, output_type: type):
        """Perform one generation (step) of the genetic algorithm.
        The step method assumes an evaluated Population and performs parent
        selection and variation (producing children).
        """
        n = self.population_size
        m = self.variation.num_parents
        t = time.time()
        parents = self.selector.select(self.population,n=n*m)
        delta = time.time()-t 
        # print(f"selection runtime {delta}")
        self.logs["selection_runtime"].append(delta)

        new_population = Population(
            [self._make_child(output_type, parents[i*m:(i+1)*m])
             for i in range(n)]
        )
        for item in self.population.logs:
            self.logs[item].append(self.population.logs[item])
        self.population = new_population


class Lexiprob(Evolver):
    """Genetic algorithm to synthesize Push programs.
    An initial Population of random Individuals is created. Each generation
    begins by evaluating all Individuals in the population. Then the current
    Population is replaced with children produced by selecting parents from
    the Population and applying VariationOperators to them.
    """

    def _make_child(self, output_type: type) -> Individual:
        # candidates - list of lists of candidates with same error vector
        global global_id
        candidates, p = self.population.nondom_set
        parent_idx = np.random.choice(
            np.arange(len(candidates)), p=p, replace=True, size=self.variation.num_parents
        )
        parents = [np.random.choice(
            candidates[i]) for i in parent_idx]
        parent_genomes = [p.genome for p in parents]
        ids = [p.ind_id for p in parents]
        ind_id=str(global_id)
        global_id+=1
        parent_hashes = [p._error_vector_bytes for p in parents]
        child_genome = self.variation.produce(parent_genomes, self.spawner)
        return Individual(child_genome, output_type,ind_id=ind_id,parent_id=ids), parent_hashes

    def _make_child_lexicase(self, output_type: type) -> Individual:
        parents = [p for p in self.selector.select(
            self.population, n=self.variation.num_parents)]
        parent_genomes = [p.genome for p in parents]
        parent_hashes = [p._error_vector_bytes for p in parents]
        child_genome = self.variation.produce(parent_genomes, self.spawner)
        return Individual(child_genome, output_type), parent_hashes

    def step(self, output_type: type):
        """Perform one generation (step) of the genetic algorithm.
        The step method assumes an evaluated Population and performs parent
        selection and variation (producing children).
        """
        lexicase_parents = []
        for _ in range(self.population_size):
            lexicase_parents += self._make_child_lexicase(output_type)[1]

        t = time.time()
        lexiprob_children = []
        lexiprob_parents = []
        for _ in range(self.population_size):
            child, parents = self._make_child(output_type)
            lexiprob_children.append(child)
            lexiprob_parents += parents
        self.logs['produce_runtime'].append(time.time()-t)

        I = 0
        T = 0
        for p in set(lexicase_parents):
            if p in set(lexiprob_parents):
                I += 1
            T += 1
        print('unique overlap: {}/{}={}'.format(I, T, I/T))
        self.logs['unique_overlap'].append((I, T))

        lexicase_count = {}
        for key in set(lexicase_parents):
            lexicase_count[key] = 0
        for p in lexicase_parents:
            lexicase_count[p] += 1

        lexiprob_count = {}
        for key in set(lexiprob_parents):
            lexiprob_count[key] = 0
        for p in lexiprob_parents:
            lexiprob_count[p] += 1

        I = 0
        T = 0
        for key in lexicase_count:
            if key in lexiprob_count:
                I += min(lexicase_count[key], lexiprob_count[key])
            T += lexicase_count[key]

        print('total overlap: {}/{}={}'.format(I, T, I/T))
        self.logs['total_overlap'].append((I, T))

        for item in self.population.logs:
            self.logs[item].append(self.population.logs[item])

        self.population = Population(lexiprob_children)


class AdaptiveCreator(ABC):
    def __init__(self, error_function=None):
        self.bandits = self.create_bandits()
        self.error_function = error_function
        self.rng = np.random.default_rng()

    @abstractmethod
    def create_bandits(self):
        pass
    
    def sample(self, population, n=1):
        return self.sample_individuals(population, self.sample_parameters(n))

    def sample_individuals(self, population, params):
        individuals = self.select_with_parameters(population, params) 
        individuals = [self.variation_with_parameters(i,p) for i,p in zip(individuals,params)]
        individuals = self.evaluate_individuals(individuals)
        for i,p in zip(individuals,params):
            self.update_with_rewards(p,i.total_error)
        return individuals

    @abstractmethod
    def sample_parameters(self, n=1):
        pass 

    @abstractmethod
    def select_with_parameters(self, population, params):
        pass 

    @abstractmethod
    def variation_with_parameters(self, parent, param):
        pass 

    def evaluate_individuals(self,individuals):
        def evaluate(indiv):
            # print(indiv)
            # print(indiv.program)
            return _eval_indiv(indiv, self.error_function)
        return [evaluate(i) for i in individuals]
    
    @abstractmethod
    def update_with_rewards(self, param, reward):
        pass 

    def initialize_creation(self, population):
        pass

def _make_child(output_type: type, parents, variation, spawner, inherit_depth=0) -> Individual:
    # print(f"Variation type: {type(variation)}")
    global global_id
    parent_genomes = [p.genome for p in parents]
    ids = [p.ind_id for p in parents]
    if isinstance(variation, VariationOperator):
        child_genome = variation.produce(parent_genomes, spawner)
    else:
        child_genome = variation(parent_genomes,spawner)
    ind_id=str(global_id)
    global_id+=1
    if len(parents)>1:
        return Individual(child_genome, output_type,ind_id=ind_id, parent_id=ids)
    else: 
        inherited_errors = parents[0].inherited_errors[:inherit_depth]
        return Individual(child_genome, output_type, inherited_errors=inherited_errors, ind_id=ind_id, parent_id = ids)

class DefaultCreator(AdaptiveCreator):
    def __init__(self, output_type, bandits, selector, variation,spawner, error_function=None, inherit_depth=0):
        super().__init__(error_function=error_function)
        self.selector = selector 
        self.variation = variation 
        self.bandits = bandits 
        self.inherit_depth=inherit_depth
        self.output_type = output_type
        self.spawner=spawner
        self.umads=[]
    def create_bandits(self):
        return [] 
    def sample_parameters(self, n=1):
        return [[self.rng.choice(bandit.sample()) for bandit in self.bandits] for _ in range(n)]
    def select_with_parameters(self,population, params):
        return self.selector(population, params)
    def variation_with_parameters(self, parent, param):
        self.umads.append(param)
        return _make_child(self.output_type, [parent], self.variation(*param), self.spawner)
    def update_with_rewards(self, param, reward):
        for p,b in zip(param,self.bandits):
            b.update(p,reward)
        return self 

class DoNothingCreator(AdaptiveCreator):
    def __init__(self, output_type, spawner, error_function=None, inherit_depth=0):
        super().__init__(error_function=error_function)
        self.inherit_depth=inherit_depth
        self.output_type = output_type
        self.spawner=spawner
        self.epsilon = False
    def create_bandits(self):
        return [] 
    def sample_parameters(self, n=1):
        return [0]*n
    def select_with_parameters(self,population, params):
        fitnesses = [i.inherited_errors for i in population]
        selected = lexicase(fitnesses,epsilon=self.epsilon, n=len(params))
        return [population[i] for i in selected]
    def variation_with_parameters(self, parent, param):
        variation = umad(0.09,0.09/(1+0.09))
        return _make_child(self.output_type, [parent], variation, self.spawner, inherit_depth=self.inherit_depth)
    def update_with_rewards(self, param, reward):
        return self 
    def initialize_creation(self, population):
        pass

def umad_variation(umad_rate=0.09):
    return umad(umad_rate,umad_rate/(1+umad_rate))

def umad_creator(error_function, output_type, spawner, selector = lexicase, low=0, high=1, step=0.1, offset_sampler=[0,0.1], step_sampler=[0.1,0.2], lr_sampler=[0.1,0.01],num_tiles=3, reward_transformation=lambda x:x):
    bandits = [MaxBanditAO(low,high,step,offset_sampler,step_sampler,lr_sampler,num_tiles,reward_transformation=reward_transformation, method="epsilon greedy")]
    def _sel(pop, params):
        fitnesses = [i.inherited_errors for i in pop]
        selected = selector(fitnesses,n=len(params))
        return [pop[i] for i in selected]
    def _var(umad_rate, *params):
        return umad_variation(umad_rate=np.exp(umad_rate))
    return DefaultCreator(output_type, bandits, _sel, _var, spawner, error_function=error_function)

def wlexicase_creator(error_function, output_type, spawner, low=0, high=1, step=0.1, offset_sampler=[0,0.1], step_sampler=[0.1,0.2], lr_sampler=[0.1,0.01],num_tiles=3, eager=True, reward_transformation=lambda x:x):
    bandits = [MaxBanditAO(low,high,step,offset_sampler,step_sampler,lr_sampler,num_tiles,reward_transformation=reward_transformation, method="epsilon greedy")]
    def _sel(pop, std):
        fitnesses = [i.inherited_errors for i in pop]
        selected = wlexicase(fitnesses, std=std)
        return pop[selected[0]]
    def _var(*params):
        return umad_variation()
    return DefaultCreator(output_type, bandits, lambda pop,params:[_sel(pop,*p) for p in params], _var, spawner, error_function=error_function)

def alpha_creator(error_function, output_type, spawner, selector = lexicase, low=0, high=1, step=0.1, offset_sampler=[0,0.1], step_sampler=[0.1,0.2], lr_sampler=[0.1,0.01],num_tiles=3, reward_transformation=lambda x:x):
    bandits = [MaxBanditAO(low,high,step,offset_sampler,step_sampler,lr_sampler,num_tiles,reward_transformation=reward_transformation, method="epsilon greedy")]
    def _sel(pop, alpha):
        fitnesses = [i.inherited_errors for i in pop]
        selected = selector(fitnesses, alpha=float(alpha))
        return pop[selected[0]]
    def _var(*params):
        return umad_variation()
    return DefaultCreator(output_type, bandits, lambda pop,params:[_sel(pop,*p) for p in params], _var, spawner, error_function=error_function)

def wlexicase_alpha_creator(error_function, low=[0,0], high=[1,1], step=[0.1,0.1], offset_sampler=[[0,0.1],[0,0.1]], step_sampler=[[0.1,0.2],[0.1,0.2]], lr_sampler=[[0.1,0.01],[0.1,0.01]],num_tiles=[3,3], reward_transformation=lambda x:x):
    bandits = [MaxBanditAO(l,h,s,o,_s,lr,n,reward_transformation=reward_transformation) for l,h,s,o,_s,lr,n in zip(low, high, step, offset_sampler, step_sampler, lr_sampler, num_tiles)]
    def _sel(pop, std, alpha):
        fitnesses = [i.inherited_errors for i in pop]
        selected = wlexicase(fitnesses, alpha=alpha, std=std)
        return [pop[s] for s in selected]
    def _var(*params):
        return umad_variation()
    return DefaultCreator(bandits, lambda pop,params:[_sel(pop,*p) for p in params], _var, error_function=error_function)

def alpha_umad_creator(error_function, low=[0,0], high=[1,1], step=[0.1,0.1], offset_sampler=[[0,0.1],[0,0.1]], step_sampler=[[0.1,0.2],[0.1,0.2]], lr_sampler=[[0.1,0.01],[0.1,0.01]],num_tiles=[3,3], reward_transformation=lambda x:x):
    bandits = [MaxBanditAO(l,h,s,o,_s,lr,n,reward_transformation=reward_transformation) for l,h,s,o,_s,lr,n in zip(low, high, step, offset_sampler, step_sampler, lr_sampler, num_tiles)]
    def _sel(pop, alpha,umad):
        fitnesses = [i.inherited_errors for i in pop]
        selected = wlexicase(fitnesses, alpha=alpha)
        return [pop[s] for s in selected]
    def _var(parent, alpha,umad_rate):
        return umad_variation(umad_rate)
    return DefaultCreator(bandits, lambda pop,params:[_sel(pop,*p) for p in params], _var, error_function=error_function)

def wlexicase_alpha_umad_creator(error_function, low=[0,0,0], high=[1,1,1], step=[0.1,0.1,0.1], offset_sampler=[[0,0.1],[0,0.1],[0,0.1]], step_sampler=[[0.1,0.2],[0.1,0.2],[0.1,0.2]], lr_sampler=[[0.1,0.01],[0.1,0.01],[0.1,0.01]],num_tiles=[3,3,3], eager=True, reward_transformation=lambda x:x):
    bandits = [MaxBanditAO(l,h,s,o,_s,lr,n,reward_transformation=reward_transformation) for l,h,s,o,_s,lr,n in zip(low, high, step, offset_sampler, step_sampler, lr_sampler, num_tiles)]
    def _sel(pop, std,alpha,umad):
        fitnesses = [i.inherited_errors for i in pop]
        selected = wlexicase(fitnesses, alpha=alpha,std=std)
        return [pop[s] for s in selected]
    def _var(parent, std,alpha,umad_rate):
        return umad_variation(umad_rate)
    return DefaultCreator(bandits, lambda pop,params:[_sel(pop,*p) for p in params], _var, error_function=error_function, eager=eager)

# def genome_creator(error_function, instruction_set, selector = eplex, low=0, high=1, step=0.1, offset_sampler=[0,0.1], step_sampler=[0.1,0.2], lr_sampler=[0.1,0.01],num_tiles=3, eager=True, reward_transformation=lambda x:x):
#     bandits = [MaxBanditAO(low,high,step,offset_sampler,step_sampler,lr_sampler,num_tiles,reward_transformation=reward_transformation)]
#     shape = [len(instruction_set),np.array(low).size]
#     rng = np.random.default_rng()
#     weights = rng.standard_normal(size=shape)
#     def _sel(pop, *params):
#         fitnesses = [i.inherited_errors for i in pop]
#         selected = selector(fitnesses)
#         return [pop[s] for s in selected]
#     def _var(parent, tile):
#         p = softmax(np.matmul(weights,tile))
#         return umad(instruction_set, parent, p=p)
#     return DefaultCreator(bandits, lambda pop,params:[_sel(pop,*p) for p in params], _var, error_function=error_function, eager=eager)


class AdaptiveGA():
    def __init__(self,
                 creator,
                 downsampler = None,
                 error_function=None,
                 spawner=None,
                 population_size: int=1000,
                 max_generations: int=300,
                 initial_genome_size: Tuple[int, int]=(30,50),
                 downsample_rate: float = 1.,
                 fitnesses=None,
                 alpha: float = 1.,
                 inherit_depth: int=0,
                 sample_size=0,):
        self.downsampler = downsampler 
        self.downsample_rate = downsample_rate
        self.creator = creator
        self.spawner=spawner
        self.error_function=error_function
        self.population_size = population_size
        self.max_generations = int(max_generations/downsample_rate)
        self.initial_genome_size = initial_genome_size
        self.population = None
        self.generation = 0
        self.best_seen = None
        self._is_solved=False
        self.alpha = alpha  # lexiprob
        self.inherit_depth = inherit_depth
        self.logs = {
            'lexiprob_runtime': [],
            'lexiprob_comparisons': [],
            'lexicase_runtime': [],
            'weighted_lexicase_runtime':[],
            'unique_overlap': [],
            'total_overlap': [],
            'eval_runtime': [],
            'produce_runtime': [],
            'total_runtime': [],
            'downsample_rate': downsample_rate,
            'alpha': alpha,
            'train_mse': [],
            'train_mae': [],
            'selection_runtime':[],
            'evaluated_solutions':[]
        }
        self.fitnesses = fitnesses
        self.sample_size=sample_size

    def init_population(self, output_type: type):
        """Initialize the population."""
        self.population = Population()
        for i in range(self.population_size):
            self.population.add(_spawn_individual(
                self.spawner, self.initial_genome_size, output_type))
        self.downsampler(self.downsample_rate)
        self.population.evaluate(self.error_function)


    def _full_step(self, output_type) -> bool:
        t = time.time()
        self.downsampler(self.downsample_rate)
        self.generation += 1
        self.creator.initialize_creation(self.population)
        
        self.population = Population([self.creator.sample(self.population)[0] for _ in range(self.population_size)])
        sampled = self.creator.bandits[0].num_sampled()
        print(sampled.reshape([10,10]).sum(axis=1))
        print(sampled[:20])

        self.logs['eval_runtime'].append(time.time()-t)
        print(f"Generation runtime {self.logs['eval_runtime'][-1]}", flush=True)

        best_this_gen = min(self.population, key=lambda x:x.total_error)
        if self.best_seen is None or best_this_gen.total_error < self.best_seen.total_error:
            self.best_seen = best_this_gen

        best_is_valid = self.best_seen.genome is not None
        
        # self.best_seen.program.pprint()
        if self.best_seen.total_error<=0:
            self._is_solved=True

        self.logs['train_mse'].append(float(best_this_gen.total_error))

        #TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        pop_stats = {"gn":round(self.generation, 3), 
                    "me": round(self.population.median_error(), 3),
                    "be":round(best_this_gen.total_error, 3),
                    "dv":round(self.population.error_diversity(), 3),
                    "best_err":round(self.best_seen.total_error, 3),
                    "best_code":escape(self.best_seen.program.root.to_code())}

        if self._is_solved:
            print("{gn}\t\t{me}\t\t{be}\t\t{dv:.3f}\t\t{best_err}\t\t{best_code}".format(
                runtime=time.time()-t,
                **pop_stats
            ), flush=True)
            return False

        t_ = time.time()
        self.logs['total_runtime'].append(time.time()-t)
        

        print("{gn}\t\t{me}\t\t{be}\t\t{dv:.3f}\t\t{best_err}\t\t{best_code}".format(
            runtime=self.logs["total_runtime"][-1],
            **pop_stats
        ), flush=True)
    
        return True

    def run(self, output_type: type) -> Individual:
        """Run the algorithm until termination."""
        self.init_population(output_type)

        print("Gen\t\tMedian\t\tBest\t\tDiv\t\tRun Best\t\tCode", flush=True)
        while self._full_step(output_type):
            eval_gens = sum(self.logs["evaluated_solutions"]) / self.population_size / self.downsample_rate
            if self.generation + eval_gens >= self.max_generations:
                break

        if self._is_solved:
            print("Solution found.", flush=True)
        else:
            print("No solution found.", flush=True)

        return self.best_seen



