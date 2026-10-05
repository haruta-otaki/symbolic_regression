# genetic programming uses variable length chromosomes (typically in tree / graphs)
# expression trees: an individual in the population (a function)
# internal nodes will be operatorss, leaves will be constant or variables operated on by functions
# post order traversal (deal with the leaves then deal with the parent)

# complicated idea: multiple population 

# symbolic regression problem: figure out function form and their coefficients
# opearators: selection, crossover, mutation

# create random population {x, range of constants}

# cross over & mutation as per videos 
# 
# cross ovver
# random 0 - 1 
# ~0.3 mutate child 
# update population 

# test population generation
# test mutation
# test crossover

# hyperparameters: 
# population size
# mutation rate
# crossover rate
# number of generations
# max depth of tree
# chance of generating "x"

import pandas as pd
import numpy as np
from Tree import Node, generate_random_tree, mutate

INITIAL_POPULATION_SIZE = 100 #500
MAX_DEPTH = 5 # 2^(5-1)= 16 leaves
OPERATORS = ["+", "-", "*", "/"]
GENERATIONS = 5 #50
CROSSOVER_RATE = 0.7
CLONE_RATE = 0.1
MUTATION_RATE = 0.2

X_PROBABILITY = 0.5
LEAF_PROBABILITY = 0.2
initial_population = []

# generate 
def generate_initial_population():
    for _ in range(INITIAL_POPULATION_SIZE):
        individual = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY)
        initial_population.append(individual)

def symbolic_regression():
    population = initial_population
    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(INITIAL_POPULATION_SIZE):
            p = np.random.random()
            if p < CROSSOVER_RATE:
                # select two parents 
                # (implement logic to prioritize higher fitness)
                parent1 = np.random.choice(population)
                parent2 = np.random.choice(population)
                # perform crossover
                child = parent1.crossover(parent2)
            elif p < CROSSOVER_RATE + CLONE_RATE:
                # select one parent
                parent = np.random.choice(population)
                # clone the parent
                child = parent.copy()
            else:
                # select one parent
                parent = np.random.choice(population)
                # perform mutation
                child = mutate(parent, OPERATORS, X_PROBABILITY)
            new_population.append(child)
            # evaluate the individual on the training data
            # calculate fitness
            
        population = new_population
    return # function

def crossover_individuals(parent1, parent2):
    return parent1.crossover(parent2)


def get_fitness(individual, x, y):
    predictions = np.array([individual.evaluate(xi) for xi in x])
    mse = np.mean((predictions - y) ** 2)
    return mse


def main():
    df = pd.read_csv('data/dataset1.csv')

    # select 80% of the data randomly for training
    train_df = df.sample(frac=0.8, random_state=42)

    # select the remaining 20% for testing using dropped indices
    test_df = df.drop(train_df.index)
    # generate_initial_population()
    # symbolic_regression()  
    individual = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY)
    individual_tree = individual.__str__()
    individual_str = individual.to_infix()
    print(f"Original Individual Tree:\n{individual_tree}")
    print(f"Original Individual: {individual_str}")
    mutated = mutate(individual, OPERATORS, X_PROBABILITY)
    mutated_tree = mutated.__str__()
    mutated_str = mutated.to_infix()
    print(f"Mutated Individual Tree:\n{mutated_tree}")
    print(f"Mutated Individual: {mutated_str}")



if __name__ == "__main__":
    main()