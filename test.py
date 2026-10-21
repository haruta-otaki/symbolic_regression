import random
import pandas as pd
import math
import numpy as np
from Tree import Node, generate_random_tree, mutate, crossover

INITIAL_POPULATION_SIZE = 500
MAX_DEPTH = 5
OPERATORS = ["+", "-", "*", "/"] 
GENERATIONS = 50
CROSSOVER_RATE = 0.7
CLONE_RATE = 0.1
MUTATION_RATE = 0.2

X_PROBABILITY = 0.5
LEAF_PROBABILITY = 0.2
TOURNAMENT_SIZE = 10
ELITE_RATIO = 0.01

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
        x_values, y_true = to_arrays(train_df, ["x"], "f(x)")
    else:
        x_values, y_true = to_arrays(train_df, ["x1", "x2", "x3"], "f(x1,x2,x3)")
    population = []
    fitnesses = []
    errors = []
    elitist_population = int(round(INITIAL_POPULATION_SIZE * ELITE_RATIO))
    print("initialize population fitness")
    # initialize fitness for initial population
    for individual in initial_population:
        evaluation = get_evaluation(individual, x_values, y_true)
        individual.fitness = evaluation[1]
        individual.error = evaluation[0]
        fitnesses.append(individual.fitness)
        errors.append(individual.error)

    print("start genetic algorithm")
    population = initial_population
    generation = 0
    current_error = float('inf')
    while generation < GENERATIONS and current_error > 1e-6:
        print(f"Generation {generation} started")
        new_population = []
        new_fitnesses = []
        new_errors = []

        elitist_indices = np.argpartition(fitnesses, (-1 * elitist_population))[(-1 * elitist_population):]
        
        for i in elitist_indices: 
            individual = population[i]
            new_population.append(individual)
            new_fitnesses.append(individual.fitness)
            new_errors.append(individual.error)

        # normalize weights so they sum to 1.0
        weights = np.array(fitnesses) / np.sum(fitnesses)

        for _ in range(INITIAL_POPULATION_SIZE - elitist_population):
            
            p = np.random.random()
            if p < CROSSOVER_RATE:
                # select two parents 
                # (implement logic to prioritize higher fitness)
                # optimization - choose index rather than node
                # tournament selection

                parent1 = tournament_selection(population, fitnesses, rng)
                parent2 = tournament_selection(population, fitnesses, rng)

                # perform crossover
                child = crossover(parent1, parent2)
            elif p < CROSSOVER_RATE + CLONE_RATE:
                # select one parent
                parent = tournament_selection(population, fitnesses, rng)
                # clone the parent
                child = parent.copy()
            else:
                # select one parent
                parent = tournament_selection(population, fitnesses, rng)
                # perform mutation
                child = mutate(parent, OPERATORS, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
            evaluation = get_evaluation(child, x_values, y_true)
            child.fitness = evaluation[1]
            child.error = evaluation[0]
            new_population.append(child)
            new_fitnesses.append(child.fitness)
            new_errors.append(child.error)
        population = new_population
        fitnesses = new_fitnesses
        errors = new_errors

        current_error = min(errors)
        print(f"Generation {generation}: Current Error = {current_error:.6f}")
        
        generation += 1
        
    max_fitness_index = np.argmax(fitnesses) 
    return population[max_fitness_index]

def tournament_selection(population, fitnesses, rng):
    indices = rng.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    winner = indices[np.argmax(np.asarray(fitnesses)[indices])]
    return population[winner]

def to_arrays(df, x, y):
    X = {}
    for v in x:
        column = df[v]                         
        array = column.to_numpy(dtype=float)   
        X[v] = array                            

    Y = df[y].to_numpy(dtype=float)
    return X, Y
    
def get_evaluation(individual, X, y):
    # temporarily ignore numpy errors
    with np.errstate(all="ignore"):
        y_pred = individual.evaluate(X)
    # check if all predictions are finite
    if not np.all(np.isfinite(y_pred)):
        return (float("inf"), 0.0)
    # calculate rmse
    error = np.sqrt(np.mean((y - y_pred) ** 2))
    # complexity penalty
    penalty = 0.001 * individual.size()
    # return fitness and error
    return (error, 1 / (1 + error + penalty))

def main():

    if DATASET == 2:
        df = pd.read_csv("data/dataset2.csv", skiprows=1, names=["x1", "x2", "x3", "f(x1,x2,x3)"])
        df["x1"] = df["x1"] / 1_000_000
        df["x2"] = df["x2"] / 1_000_000
    else:
        df = pd.read_csv("data/dataset1.csv")

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
    x_true_values, y_true_values = to_arrays(test_df, ["x"], "f(x)")
    if DATASET == 1:
        x_true_values, y_true_values = to_arrays(test_df, ["x"], "f(x)")
    else:
        x_values, y_true = to_arrays(train_df, ["x1", "x2", "x3"], "f(x1,x2,x3)")

        
    error, fitness = get_evaluation(most_fit, x_true_values, y_true_values)

    print(f"Test RMSE: {error}")
    print(f"Test Fitness: {fitness}")

    
if __name__ == "__main__":
    main()
