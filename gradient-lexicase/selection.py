from copy import deepcopy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from mutation import *
from typing import Tuple
import random

def process_idx(idx, length):
        assert type(idx)==tuple or type(idx)==int
        if not type(idx)==tuple:
            if idx>=0:
                idx = (idx,idx+1)
            else:
                idx =  (max(0,length+idx), length)
        assert idx[0]>=0 and idx[1]<=length, "Index out of bounds"
        return idx

def threshold(tensor: torch.Tensor, shuffle: torch.Tensor, threshold_value, filter_op = lambda x,y:x>=y):
        """
        Given the index (or a range of indices) of the generations whose results to filter by, and a shuffle of those flattened results, returns the 
        index of the first test case that does not pass the threshold (float or tensor). Returns the numger of test cases if no such test case is found. Optionally specify
        function with which to compare tensor to threshold.
        """ 
        assert shuffle.size()==torch.Size([tensor.size(-1)]), "Shuffle must be size [batch_size]"

        #Get shuffled cases
        cases = tensor.index_select(-1,shuffle)
        if isinstance(threshold_value,torch.Tensor):
            threshold_value = threshold_value.index_select(-1,shuffle)
        # print('cases')
        # print(cases)
        # print('threshold_value')
        # print(threshold_value)
        #Cases that pass threshold are assigned 0. Cases that don't are assigned -1
        # print("LE")
        # print(cases<=threshold_value.index_select(-1,shuffle))
        passed = torch.where(filter_op(cases, threshold_value), 0, -1)
        # print("passed")
        # print(passed)
        #Indices of test cases where the individual failed
        
        fail_min, fail_indices = passed.min(dim=-1)
        fail_indices[fail_min==0] = cases.size(-1)

        # if random.random()<0.005:
        #     print(cases)
        #     print(passed)
        #     print(fail_indices)
        return fail_indices


class TestCases(Dataset):
    def __init__(self, noise_table = None, informed = False, train=True, length = None):
        
        
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.data = torchvision.datasets.CIFAR10(root = './data',train=train,download=True,transform=self.transform)
        

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if informed: 
            assert noise_table, "Must provide a noise table if informed is True"
            assert noise_table.noise_type == 'uniform', "Must provide a uniform noise table"
            assert len(noise_table.random_noise) >= len(self.data), \
            "Noise block doesn't have enough datapoints. Need: " + str(len(self.data)) + ", have: " + str(len(noise_table.random_noise))
            self.weights = torch.ones(len(self.data))
            self.noise_table = noise_table
            
        self.shuffled_indices = torch.randperm(len(self.data))
        
        self.informed = informed
        self.length = length

    def __len__(self):
        return self.length if self.length else len(self.data)
    
    def __getitem__(self, idx):
        return self.data[self.shuffled_indices[idx]] if self.informed else self.data[idx]
    
    def update_indices(self):
        #Returns the indices of the weighted shuffle. Uses gumbel max trick to sort test cases using logit weights
        idx = self.noise_table.get_seed(len(self.data))
        uniform = self.noise_table.get(idx, len(self.data))
        gumbel = uniform.log().neg_().log_().neg_()
        self.shuffled_indices = self.weights.add(gumbel).argsort(descending=True)
    
    def update_weights(self):
        raise NotImplementedError
    
class FitnessResults():
    def __init__(self, batch_size, device='cpu'):
        #Results in a tensor of shape [num-ancestors, batch_size]
        self.results = torch.empty((0,batch_size), device=device)
        self.batch_size = batch_size
        self.decay = None

    def track_scalar(self, decay=0.8):
        self.decay = decay
        self.scalar_fitness=0
        return self
    
    def untrack_scalar(self):
        self.decay=None
        return self

    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return self.results[idx]
    
    def deepcopy(self):
        dup = FitnessResults(self.batch_size)
        dup.results = self.results.clone()
        if self.decay:
            dup.decay=self.decay
            dup.scalar_fitness=self.scalar_fitness
        return dup
    
    def threshold(self, idx, shuffle: torch.Tensor, threshold_value: float, filter_op = lambda x,y:x>=y, later_gens_first=True):
        """
        Given the index (or a range of indices) of the generations whose results to filter by, and a shuffle of those flattened results, returns the 
        index of the first test case that does not pass the threshold. Returns the numger of test cases if no such test case is found. Optionally specify
        function with which to compare tensor to threshold.
        """ 
        #Convert idx to tuple
        idx = process_idx(idx,len(self.results))

        #Convert shuffle from a list of shuffles to a flat shuffle
        if shuffle.dim()==2:
            range = torch.arange(0,shuffle.numel(),shuffle.size(1)).to(self.device)
            range = range.flip(0) if later_gens_first else range
            shuffle+=range.unsqueeze(-1).expand(shuffle.size())
            shuffle=shuffle.flatten()
        else:
            shuffle = shuffle.flip(0) if later_gens_first else shuffle
        
        #Check size of shuffle
        assert shuffle.size()==torch.Size([self.batch_size*(idx[1]-idx[0])]), "Shuffle must be size [batch_size]"

        #Get shuffled cases
        cases = self.results[idx[0]: idx[1]].flatten()
        fail_indices = threshold(cases.unsqueeze(0), shuffle, threshold_value)
        return fail_indices.item()
    
    
    def add_result(self, result: torch.Tensor):
        assert len(result)==self.batch_size if result.dim()==1 else result.size(dim=1)==self.batch_size,\
            f"Input result must be of shape [batch_size]: result={result.size()}, batch_size={self.batch_size}"
        if result.dim()==1:
            self.results = torch.cat((self.results,result.reshape(1,self.batch_size)))
        else:
            self.results = torch.cat((self.results,result))

        if self.decay:
            if self.scalar_fitness:
                self.scalar_fitness = self.decay*self.scalar_fitness + (1-self.decay)*result.mean()
            else:
                self.scalar_fitness = result.mean()
            

    def assert_length(self, length):
        self.results = self.results[-length:]

    def __str__(self) -> str:
        return str(self.results)

    def aggregate_results(self, idx: Tuple, factor: int):
        #Convert idx to tuple
        idx = self.process_idx(idx)
        assert (idx[1]-idx[0])%factor==0, f"Factor must divide number of indices: factor={factor}, indices={idx}"

        #Regroup each factor elements together, and take the average
        middle = self.results[idx[0]:idx[1]].reshape(-1, self.batch_size, factor).mean(dim=-1)

        #Recombine with original tensor
        self.results = torch.concat((self.results[:idx[0]], middle, self.results[idx[1]:]))

class Individual:
    def __init__(self, net, momentum:float, batch_size:int, device='cpu'):
        assert 0<=momentum<=1, f"Momentum must be between 0 and 1: momentum={momentum}"
        self.net = net #resnet_binary(dataset='cifar10', depth = 1)
        self.results = FitnessResults(batch_size, device=device)
        self.velocity = None 
        self.momentum = momentum 

    def mutate(self, noise_table: NoiseTable, stdev=0.005):
        m = Mutation(self.net.parameters())
        t = noise_table.get(noise_table.get_seed(m.total_size), m.total_size)
        self.update_velocity(t)
        m.add_parameters((self.momentum*self.velocity+t)*stdev, xavier=False)
        return self
        
    def set_momentum(self, momentum:float):
        self.momentum = momentum 
        return self

    def update_velocity(self, mutation):
        if self.velocity is not None:
            self.velocity = self.velocity*self.momentum+mutation
        else:
            self.velocity = mutation

    def deepcopy(self, net):
        i = Individual(net,self.momentum,self.results.batch_size)
        i.net.load_state_dict(deepcopy(self.net.state_dict()))
        i.results=self.results.deepcopy()
        if self.velocity is not None:
            i.velocity = self.velocity.clone()
        return i

    def loss(self, inputs, target, loss_fn):
        return loss_fn(self.net(inputs),target)
    
    def parameters(self):
        return self.net.parameters()

class Selection():
    def __init__(self, selection_type = 'lexicase', threshold_value=0.5, minibatch_size = 1, filter_op = lambda x,y: x<=y, use_ancestral = False, truncation_size=0, tournament_size=10, device='cpu'):
        assert selection_type.lower() in ['lexicase', 'tournament', 'epsilon lexicase', 'batch lexicase', 'batch epsilon lexicase', 'fitness proportional', 'elite', 'truncation'],\
            f"Selection type not implemented: type={selection_type}"
        assert not use_ancestral, "Ancestral fitness not implemented yet"
        self.selection_type = selection_type.lower()
        self.threshold_value = threshold_value
        self.minibatch_size = minibatch_size
        self.filter_op = filter_op
        self.use_ancestral = use_ancestral
        self.truncation_size = truncation_size
        self.tournament_size = tournament_size
        self.device = device
    
    def select_parents(self, pop: list[Individual], net_fn, ):
        #Pop is a list of (net, result, parents) tuples
        parents = self.select_by_type(pop)
        # print(parents)
        return [pop[i].deepcopy(net_fn()) for i in parents]
        
        
    def select_by_type(self, pop: list[Individual]):
        if self.selection_type=='lexicase':
            return [self.lexicase(pop) for _ in pop]
        elif self.selection_type=='epsilon lexicase':
            return [self.epsilon_lexicase(pop) for _ in pop]
        elif self.selection_type == 'batch lexicase':
            return [self.batch_lexicase(pop) for _ in pop]
        elif self.selection_type == 'batch epsilon lexicase':
            return [self.batch_epsilon_lexicase(pop) for _ in pop]
        elif self.selection_type == 'fitness proportional':
            return self.fitness_proportional(pop)
        elif self.selection_type == 'elite':
            return self.elite(pop)
        elif self.selection_type =='tournament':
            return [self.tournament(pop) for _ in pop]
        elif self.selection_type == 'truncation':
            return self.truncation(pop)
        
    def tournament(self, pop):
        tournament = random.sample(pop,self.tournament_size)
        return self.elite(tournament)[0]

    def truncation(self, pop):
        results = torch.vstack([i.results.results[-1] for i in pop])
        avg_results = results.mean(-1).argsort()
        truncation_size = int(self.truncation_size*len(pop))
        return [avg_results[random.randint(0,truncation_size)].item() for _ in pop]

    def elite(self, pop):
        results = torch.vstack([i.results.results[-1] for i in pop])
        elite = results.mean(-1).argmin().item()
        return [elite]*len(pop)
        
    def fitness_proportional(self, pop):
        assert all(i.results.decay for i in pop)
        fitnesses = torch.tensor([1/i.results.scalar_fitness for i in pop]).softmax(-1)
        #Multiple roulette wheels
        selected = fitnesses.multinomial(len(pop), replacement=True)
        return selected
        
        
    def multi_argmax(self, tensor: torch.Tensor) -> torch.Tensor:
        #Returns a 2d tensor of all the indices where a maximal value is at.
        m = tensor.max()
        return torch.argwhere(torch.where(tensor==m,1,0)).flatten()
        
    def lexicase(self, pop: list[Individual]):
        #Lexicase is batch lexicase with a batch size of 1. Threshold is 0.5 to distinguish between 0 (unsolved) and 1 (solved)
        assert self.minibatch_size==1, "Lexicase requires minibatch size 1"
        assert 0<self.threshold_value<1, "Lexicase threshold must be between 0 and 1"
        return self.batch_lexicase(pop)
    
    def epsilon_lexicase(self, pop: list[Individual]):
        #Tensor implementation of epsilon lexicase selection. 
        #Stack results of the latest fitness evaluation 
        results = torch.vstack([i.results.results[-1] for i in pop])
        #Standardize test case shuffling
        shuffle = torch.randperm(results.size(-1)).to(self.device)
        # print("shuffle")
        # print(shuffle)
        
        #Compute threshold = max - mean absolute deviation across population for each test ase
        threshold_value = results.min(0)[0]+(results-results.mean(0)).abs().median(0)[0]
        # print("threshold")
        # print(threshold_value)


        fail_indices = threshold(results, shuffle, threshold_value, filter_op=self.filter_op)
        # print('fail_indices')
        # print(fail_indices)
        survivors = self.multi_argmax(fail_indices)
        
        # print("survivors")
        # print(survivors)
        return survivors.index_select(0,torch.randint(0,len(survivors),(1,), device=self.device)).item()
    
    def batch_lexicase(self, pop: list[Individual]):
        #Stack results of the latest fitness evaluation 
        results = torch.vstack([i.results.results[-1] for i in pop])
        assert results.size(-1)%self.minibatch_size==0, "minibatch size must divide batch size"
        num_minibatches = int(results.size(-1)/self.minibatch_size)
        

        shuffle = torch.randperm(num_minibatches, device=self.device)
        cases = results.reshape(results.size()[:-1]+ (num_minibatches,self.minibatch_size)).mean(-1).squeeze(-1)
        fail_indices = threshold(cases, shuffle, self.threshold_value, filter_op=self.filter_op)
        survivors = self.multi_argmax(fail_indices)
        return survivors.index_select(0,torch.randint(0,len(survivors),(1,), device=self.device)).item()
    
    def batch_epsilon_lexicase(self, pop):
        #Tensor implementation of batched epsilon lexicase selection. 
        #Stack results of the latest fitness evaluation 
        results = torch.vstack([i.results.results[-1] for i in pop])

        
        assert results.size(-1)%self.minibatch_size==0, "minibatch size must divide batch size"
        num_minibatches = int(results.size(-1)/self.minibatch_size)

        #Standardize test case shuffling
        shuffle = torch.randperm(num_minibatches, device=self.device)
        cases = results.reshape(results.size()[:-1]+ (num_minibatches,self.minibatch_size)).mean(-1).squeeze(-1)
        
        #Compute threshold = min + mean absolute deviation across population for each test ase
        threshold_value = cases.min(0)[0]+(cases-cases.mean(0)).abs().median(0)[0]

        fail_indices = threshold(cases, shuffle, threshold_value, filter_op=self.filter_op)
        
        survivors = self.multi_argmax(fail_indices)
        return survivors.index_select(0,torch.randint(0,len(survivors),(1,), device=self.device)).item()

    
        




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
    case_count=[]

    for _ in range(n):
        count=0
        candidates = range(popsize)
        for i in range(inherit_depth):
            if len(candidates) <= 1:
                break
            ordering = rng.permutation(num_cases)
            for case in ordering:
                if len(candidates) <= 1:
                    break
                count+=1
                errors_this_case = [error_matrix[ind][i][case] for ind in candidates]
                best_val_for_case = min(errors_this_case)+ep[i][case]
                candidates = [ind for ind in candidates if error_matrix[ind][i][case] <= best_val_for_case]
        case_count.append(count)
        selected.append(rng.choice(candidates))

    
    values, counts = np.unique(selected,return_counts=True)

    p=counts/np.sum(counts)
    p=p**alpha 
    p=p/np.sum(p)

    selected = rng.choice(values,p=p,replace=True, size=n)

    return [rng.choice(error_groups[i][1]) for i in selected], case_count


def dalex(fitnesses, n=1, alpha=1, std=np.array(1).reshape([1,1,1]), offset=np.array(0).reshape([1,1,1]), distribution="normal", epsilon=False):
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

    # error_matrix = (error_matrix-(np.mean(error_matrix,axis=0)))/np.std(error_matrix,axis=0)
    
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

    return [rng.choice(error_groups[i][1]) for i in selected], [inherit_depth*num_cases]*n



    







        
