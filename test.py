# complicated idea: multiple population 

# create random population {x, range of constants}

# cross over & mutation as per videos 
# 
# cross ovver
# random 0 - 1 
# ~0.3 mutate child 
# update population 

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from Tree import Node, generate_random_tree, mutate, crossover

INITIAL_POPULATION_SIZE = 100 #500
MAX_DEPTH = 5 # 2^(5-1)= 16 leaves
OPERATORS = ["+", "-", "*", "/"]
GENERATIONS = 5 #50
CROSSOVER_RATE = 0.7
CLONE_RATE = 0.1
MUTATION_RATE = 0.2
CROSS_OVER_MAX_DEPTH = 8

X_PROBABILITY = 0.5
LEAF_PROBABILITY = 0.2

# generate 
def generate_initial_population():
    initial_population = []
    for _ in range(INITIAL_POPULATION_SIZE):
        individual = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY)
        initial_population.append(individual)
    return initial_population

def symbolic_regression(train_df, initial_population):
    x_values = train_df['x'].tolist()
    y_true = train_df['f(x)'].tolist()
    population = []
    fitnesses = []
    # initialize fitness for initial population
    for individual in initial_population:
        individual.fitness = get_fitness(individual, x_values, y_true)
        fitnesses.append(individual.fitness)

    population = initial_population
    for generation in range(GENERATIONS):
        new_population = []
        new_fitnesses = []
        for _ in range(INITIAL_POPULATION_SIZE):
            p = np.random.random()
            if p < CROSSOVER_RATE:
                # select two parents 
                # (implement logic to prioritize higher fitness)
                parent1 = np.random.choices(population, weights=fitnesses, k=1)[0]
                parent2 = np.random.choices(population, weights=fitnesses, k=1)[0]

                # perform crossover
                child = crossover(parent1.copy(), parent2.copy(), OPERATORS, X_PROBABILITY, CROSS_OVER_MAX_DEPTH)
            elif p < CROSSOVER_RATE + CLONE_RATE:
                # select one parent
                parent = np.random.choices(population, weights=fitnesses, k=1)[0]
                # clone the parent
                child = parent.copy()
            else:
                # select one parent
                parent = np.random.choices(population, weights=fitnesses, k=1)[0]
                # perform mutation
                child = mutate(parent.copy(), OPERATORS, X_PROBABILITY)
            child.fitness = get_fitness(child, x_values, y_true)
            new_population.append(child)
            new_fitnesses.append(child.fitness)
        population = new_population
        fitnesses = new_fitnesses
    return # function

def get_fitness(individual, x_values, y_true):
    y_pred = []
    for x in x_values:
        y_pred.append(individual.evaluate(x))
    return r2_score(y_true, y_pred)

def main():
    df = pd.read_csv('data/dataset1.csv')

    # select 80% of the data randomly for training
    train_df = df.sample(frac=0.8, random_state=42)

    # select the remaining 20% for testing using dropped indices
    test_df = df.drop(train_df.index)

    initial_population = generate_initial_population()
    symbolic_regression(train_df, initial_population)  

    # testing: 

    # individual_1 = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY)
    # individual_1_tree = individual_1.__str__()
    # individual_1_str = individual_1.to_infix()
    # print(f"Original Individual Tree:\n{individual_1_tree}")
    # print(f"Original Individual: {individual_1_str}")
    # individual_2 = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY)
    # individual_2_tree = individual_2.__str__()
    # individual_2_str = individual_2.to_infix()
    # print(f"Original Individual Tree:\n{individual_2_tree}")
    # print(f"Original Individual: {individual_2_str}")
    # # mutated = mutate(individual, OPERATORS, X_PROBABILITY)
    # # mutated_tree = mutated.__str__()
    # # mutated_str = mutated.to_infix()
    # # print(f"Mutated Individual Tree:\n{mutated_tree}")
    # # print(f"Mutated Individual: {mutated_str}")
    # crossover_node = crossover(individual_1, individual_2, CROSS_OVER_MAX_DEPTH)
    # crossover_tree = crossover_node.__str__()
    # crossover_str = crossover_node.to_infix()
    # print(f"Crossover Individual Tree:\n{crossover_tree}")
    # print(f"Crossover Individual: {crossover_str}")



if __name__ == "__main__":
    main()