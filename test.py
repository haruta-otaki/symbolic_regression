import sympy as sp
import random
import pandas as pd
import math
import numpy as np
from Tree import Node, generate_random_tree, mutate, crossover

INITIAL_POPULATION_SIZE = 500
MAX_DEPTH = 5
BINARY_OPERATORS = ["+", "-", "*", "/"]
UNARY_OPERATORS = ["sin", "log", "exp"]

GENERATIONS = 20
CROSSOVER_RATE = 0.7
CLONE_RATE = 0.1
MUTATION_RATE = 0.2

X_PROBABILITY = 0.5
LEAF_PROBABILITY = 0.2
UNARY_PROBABILITY = 0.2
TOURNAMENT_SIZE = 5
ELITE_RATIO = 0.01

DATASET = 3

if DATASET == 1 or DATASET == 3:
    VARIABLES = ["x"]
    REAL_CONSTANTS = False
else:
    VARIABLES = ["x1", "x2", "x3"]
    REAL_CONSTANTS = True


    # generate
def generate_initial_population():
    initial_population = []
    for _ in range(INITIAL_POPULATION_SIZE):
        if DATASET == 3:
            individual = generate_random_tree(BINARY_OPERATORS, UNARY_OPERATORS, UNARY_PROBABILITY, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
        else: 
            individual = generate_random_tree(BINARY_OPERATORS, UNARY_OPERATORS, 0.0, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)

        initial_population.append(individual)
    return initial_population

def symbolic_regression(train_df, initial_population):
    rng = np.random.default_rng()
    if DATASET == 1 or DATASET == 3:
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
                if DATASET == 3: 
                    child = mutate(parent, BINARY_OPERATORS, UNARY_OPERATORS, UNARY_PROBABILITY, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
                else: 
                    child = mutate(parent, BINARY_OPERATORS, UNARY_OPERATORS, 0.0, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
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
    if DATASET == 1:
        # linear scaling: best a + b * prediction, then RMSE
        var_p = np.var(y_pred)
        # no variance
        if var_p < 1e-12:
            coefficient = 0.0
        else:
            coefficient = np.mean((y_pred - y_pred.mean()) * (y - y.mean())) / var_p
        individual.shift = y.mean() - coefficient * y_pred.mean()
        individual.coefficient = coefficient
        # calculate rmse
        error = np.sqrt(np.mean((y - (individual.shift + coefficient * y_pred)) ** 2))
        # complexity penalty
        penalty = 0.01 * individual.size()
        # return fitness and error
        return (error, 1 / (1 + error + penalty))
    elif DATASET == 2:
        # f is rounded to 2 decimals, so a printed 0.00 is between 0 and 0.005
        y = np.maximum(y, 0.001)
        # RMSLE needs positive predictions
        if np.any(y_pred <= 0):
            return (float("inf"), 0.0)
        log_gap = np.log(y) - np.log(y_pred)      # log of (true / predicted) per row
        log_c = np.mean(log_gap)                  # best multiplier, in log form 
        individual.shift = 0.0
        individual.coefficient = np.exp(log_c);         
        error = np.sqrt(np.mean((log_gap - log_c) ** 2))
        penalty = 0.01 * individual.size()
        return (error, 1 / (1 + error + penalty))
    else: 
        individual.shift = 0.0
        individual.coefficient = 1.0;     
        error = np.sqrt(np.mean((y - (individual.shift + individual.coefficient * y_pred)) ** 2))
        # complexity penalty
        penalty = 0.01 * individual.size()
        # return fitness and error
        return (error, 1 / (1 + error + penalty))

def get_error(individual, X, y, df_copy):
    with np.errstate(all="ignore"):
        y_pred = individual.coefficient * individual.evaluate(X) + individual.shift
    # check if all predictions are finite
    if not np.all(np.isfinite(y_pred)):
        return (float("inf"), 0.0)

    if DATASET == 2:
        df_copy["pred"] = y_pred
        df_copy["abs_error"] = np.abs(df_copy["f(x1,x2,x3)"] - df_copy["pred"])


        # 3. sort by error, biggest first, and show the top 5
        print(df_copy.sort_values("abs_error", ascending=False).head(20))

    rel = np.abs(y - y_pred) / y
    print(f"Median relative error: {100 * np.median(rel):.3f}%")
    # calculate rmse
    error = np.sqrt(np.mean((y - y_pred) ** 2))
    # complexity penalty
    penalty = 0.01 * individual.size()
    # return fitness and error
    return (error, 1 / (1 + error * (1 + penalty)))

def main():
    if DATASET == 2:
        df = pd.read_csv(f"data/dataset{DATASET}.csv", skiprows=1, names=["x1", "x2", "x3", "f(x1,x2,x3)"])
        before = len(df)
        df = df[df["x3"] >= 0.1]
        print(f"removed {before - len(df)} rows with x3 <= 0.1 (too imprecise after rounding)")

    else:
        df = pd.read_csv("data/dataset1.csv")
    
    print("Dataset:", DATASET)
    print("Columns:", [repr(col) for col in df.columns])
    # print(df.head())


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
    expr = sp.sympify(most_fit.to_infix())
    print("Simplified:", sp.simplify(most_fit.coefficient * expr + most_fit.shift))
    
    # testing:
    if DATASET == 1 or DATASET == 3:
        x_true_values, y_true_values = to_arrays(test_df, ["x"], "f(x)")
    else:
        x_true_values, y_true_values = to_arrays(test_df, ["x1", "x2", "x3"], "f(x1,x2,x3)")
        
    df_copy = test_df.copy()
    error, fitness = get_error(most_fit, x_true_values, y_true_values, df_copy)
    print(f"Test Coef: {most_fit.coefficient}")
    print(f"Test Error: {error}")
    print(f"Test Fitness: {fitness}")
    
if __name__ == "__main__":
    main()
