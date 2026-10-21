# complicated idea: multiple population 

# create random population {x, range of constants}

# cross over & mutation as per videos 
# 
# cross ovver
# random 0 - 1 
# ~0.3 mutate child 
# update population 

import pandas as pd
import math
import numpy as np
from sklearn.metrics import root_mean_squared_error
from Tree import Node, generate_random_tree, mutate, crossover

INITIAL_POPULATION_SIZE = 500
MAX_DEPTH = 6
OPERATORS = ["+", "-", "*", "/"]
GENERATIONS = 50
CROSSOVER_RATE = 0.7
CLONE_RATE = 0.1
MUTATION_RATE = 0.2

X_PROBABILITY = 0.5
LEAF_PROBABILITY = 0.3

DATASET = 2

if DATASET == 1:
    VARIABLES = ["x"]
    REAL_CONSTANTS = False
else:
    VARIABLES = ["x1", "x2", "x3"]
    REAL_CONSTANTS = True

# generate 
def generate_initial_population():
    initial_population = []
    for _ in range(INITIAL_POPULATION_SIZE):
        individual = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
        initial_population.append(individual)
    return initial_population

def symbolic_regression(train_df, initial_population):
    rng = np.random.default_rng()
    if DATASET == 1:
        x_values = train_df["x"].tolist()
        y_true = train_df["f(x)"].tolist()
    else:
        x_values = train_df[["x1", "x2", "x3"]].to_dict("records")
        y_true = train_df["f(x1,x2,x3)"].tolist()
    population = []
    fitnesses = []
    print("initialize population fitness")
    # initialize fitness for initial population
    for individual in initial_population:
        individual.fitness = get_fitness(individual, x_values, y_true)
        fitnesses.append(individual.fitness)

    print("start genetic algorithm")
    population = initial_population
    for generation in range(GENERATIONS):
        print(f"Generation {generation} started")

        best_index = np.argmax(fitnesses)
        best_individual = population[best_index].copy()

        new_population = [best_individual]
        new_fitnesses = [best_individual.fitness]
        
        for _ in range(INITIAL_POPULATION_SIZE - 1):
            # normalize weights so they sum to 1.0
            weights = np.array(fitnesses) / np.sum(fitnesses)
            p = np.random.random()
            if p < CROSSOVER_RATE:
                # select two parents 
                # (implement logic to prioritize higher fitness)

                parent1 = rng.choice(population, p=weights)
                parent2 = rng.choice(population, p=weights)

                # perform crossover
                child = crossover(parent1, parent2)
            elif p < CROSSOVER_RATE + CLONE_RATE:
                # select one parent
                parent = rng.choice(population, p=weights)
                # clone the parent
                child = parent.copy()
            else:
                # select one parent
                parent = rng.choice(population, p=weights)
                # perform mutation
                child = mutate(parent.copy(), OPERATORS, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
            child.fitness = get_fitness(child, x_values, y_true)
            new_population.append(child)
            new_fitnesses.append(child.fitness)
        population = new_population
        fitnesses = new_fitnesses

        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness:.6f}")
        
    max_fitness_index = np.argmax(fitnesses) 
    return population[max_fitness_index]

def get_fitness(individual, x_values, y_true):
    y_pred = []

    try:

        for x in x_values:
            prediction = individual.evaluate(x)

            #for infinity or non-number
            if not math.isfinite(prediction):
                return 0

            y_pred.append(prediction)
        
        error = root_mean_squared_error(y_true, y_pred)

        #for infinity or non-number
        if not math.isfinite(error):
            return 0
        
        complexity_penalty = 0.001 * individual.size()
        return 1 / (1 + error + complexity_penalty)
        
    except (ValueError, OverflowError, ZeroDivisionError):
        return 0


def main():
    df = pd.read_csv(f"data/dataset{DATASET}.csv")

    print("Dataset:", DATASET)
    print("Actual columns:", [repr(col) for col in df.columns])
    print(df.head())

    # select 80% of the data randomly for training
    train_df = df.sample(frac=0.8, random_state=42)

    # select the remaining 20% for testing using dropped indices
    test_df = df.drop(train_df.index)

    print("generate initial population")
    initial_population = generate_initial_population()
    print("start symbolic regression")
    most_fit = symbolic_regression(train_df, initial_population)  
    most_fit_tree = most_fit.__str__()
    most_fit_str = most_fit.to_infix()
    print(f"Original Individual Tree:\n{most_fit_tree}")
    print(f"Original Individual: {most_fit_str}")
    
    # testing: 

    if DATASET == 1:
        x_true_values = test_df["x"].tolist()
        y_true_values = test_df["f(x)"].tolist()
    else:
        x_true_values = test_df[["x1", "x2", "x3"]].to_dict("records")
        y_true_values = test_df["f(x1,x2,x3)"].tolist()

    test_fitness = get_fitness(most_fit, x_true_values, y_true_values)

    print(f"Test Fitness: {test_fitness}")
    

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
