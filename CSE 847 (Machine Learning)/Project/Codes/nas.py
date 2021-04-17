# !curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
# !git clone https://github.com/google-research/nasbench
# !pip install ./nasbench

# Initialize the NASBench object which parses the raw data into memory (this
# should only be run once as it takes up to a few minutes).
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nasbench import api

# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = api.NASBench('nasbench_only108.tfrecord')

# Standard imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import random

# Useful constants for NAS-Bench-101
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

"""## Basic usage"""

# Query a cell from the dataset.
cell = api.ModelSpec(
  matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
          [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
          [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
          [0, 0, 0, 0, 0, 0, 0]],   # output layer
  # Operations at the vertices of the module, matches order of matrix.
  ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Querying multiple times may yield different results. Each cell is evaluated 3
# times at each epoch budget and querying will sample one randomly.
data = nasbench.query(cell)
for k, v in data.items():
  print('%s: %s' % (k, str(v)))

def fitness(solution):
  # Fitness metric for GA
  spec = api.ModelSpec(matrix=solution.matrix, ops=solution.ops)
  if nasbench.is_valid(spec):
    metrics = nasbench.query(spec)
    return metrics['validation_accuracy']
  else:
    return 0

def call_counter(fn):
    # hierarchical function to count number of evaluation calls
    def helper(*args, **kwargs):
        helper.calls += 1
        cur_val = fn(*args, **kwargs)
        if(cur_val > helper.best_val):
          helper.best_val = cur_val
          print('Best Val:{}, Func Eval:{}'.format(helper.best_val, helper.calls))
        
        return cur_val
    helper.__name__ = fn.__name__
    helper.calls = 0
    helper.best_val = float('-inf')
    return helper

# Associating the function counter with the fitness function
fitness = call_counter(fitness)


class Candidate():
  # Class for representing every candidate solution
  def __init__(self, matrix, ops):
    self.matrix = matrix
    self.ops = ops


def mutation(solution, mutation_rate=1.0):
  # function for performing mutation
  while True:
    new_matrix = solution.matrix.copy()
    new_ops = solution.ops.copy()

    # In expectation, V edges flipped (note that most end up being pruned).
    edge_mutation_prob = mutation_rate / NUM_VERTICES
    for src in range(0, NUM_VERTICES - 1):
      for dst in range(src + 1, NUM_VERTICES):
        if random.random() < edge_mutation_prob:
          new_matrix[src, dst] = 1 - new_matrix[src, dst]
       
    # In expectation, one op is resampled.
    op_mutation_prob = mutation_rate / OP_SPOTS
    for ind in range(1, NUM_VERTICES - 1):
      if random.random() < op_mutation_prob:
        available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
        new_ops[ind] = random.choice(available)
    
    new_spec = api.ModelSpec(new_matrix, new_ops)
    if nasbench.is_valid(new_spec):
      return Candidate(new_matrix, new_ops)


def crossover(parent1, parent2):
  # function for performing crossover over two parent chromosomes
  child1_mat = np.zeros((NUM_VERTICES, NUM_VERTICES),dtype=int)
  child1_ops = []
  child2_mat = np.zeros((NUM_VERTICES, NUM_VERTICES),dtype=int)
  child2_ops = []
  cross_point = np.random.randint(NUM_VERTICES-2)+1
  
  for i in range(cross_point):
    child1_ops.append(parent1.ops[i])
    child1_mat[:,i] = parent1.matrix[:,i]
    child2_ops.append(parent2.ops[i])
    child2_mat[:,i] = parent2.matrix[:,i]

  for i in range(cross_point, NUM_VERTICES):
    child1_ops.append(parent2.ops[i])
    child1_mat[:,i] = parent2.matrix[:,i]
    child2_ops.append(parent1.ops[i])
    child2_mat[:,i] = parent1.matrix[:,i]

  child1 = Candidate(child1_mat, child1_ops)
  child2 = Candidate(child2_mat, child2_ops)
  return child1, child2


def selection(population, objective):
    # function to perform tournament selection
    K = 5
    num_pop = len(population)
    perm = np.random.permutation(num_pop)
    pop_comb = perm[0:K]
    tournament_fit = np.zeros(K)

    # creating tournament population
    for i in range(K):
        tournament_fit[i] = objective[pop_comb[i]]

    # declaring winners
    idx = np.argsort(tournament_fit)
    parent_id1 = idx[0] 
    parent_id2 = idx[1]
    return parent_id1, parent_id2


def initialization(pop_size):
  # function for initialization
    population = []
    for pop_no in range(pop_size):
      matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
      matrix = np.triu(matrix, 1)
      ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
      ops[0] = INPUT
      ops[-1] = OUTPUT
      population.append(Candidate(matrix, ops))

    return population

def GA(pop_size=50, num_gen=100, cross_mut_prob=0.5):
  # main driver function for GA
  population = initialization(pop_size)
  obj_values = np.zeros(pop_size)
  avg_values = np.zeros(num_gen)

  for pop_no in range(pop_size):
    # calculate fitness for all the chromosomes
    obj_values[pop_no] = fitness(population[pop_no])

  for gen_no in range(num_gen):
    avg_values[gen_no] = np.mean(obj_values)
 
    for i in range(max_cross):
      if(np.random.rand()<cross_mut_prob):

        # parent selection
        parent_id1, parent_id2 = selection(population, obj_values)

        # crossover
        child_1, child_2 = crossover(population[parent_id1], population[parent_id2])

        # mutation
        child_1 = mutation(child_1)
        child_2 = mutation(child_2)

        # child fitness computation
        obj_1 = fitness(child_1)
        obj_2 = fitness(child_2)
        
        # child replaces the worst solution if applicable
        if(obj_1 > min(obj_values)):
            idx = np.argmin(obj_values)
            population[idx] = child_1
            obj_values[idx] = obj_1
            
        if(obj_2 > min(obj_values)):
            idx = np.argmin(obj_values)
            population[idx] = child_2
            obj_values[idx] = obj_2


# calling GA
GA()

