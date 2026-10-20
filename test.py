import pandas as pd
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

# generate 
def generate_initial_population():
    initial_population = []
    for _ in range(INITIAL_POPULATION_SIZE):
        individual = generate_random_tree(OPERATORS, MAX_DEPTH, LEAF_PROBABILITY, X_PROBABILITY)
        initial_population.append(individual)
    return initial_population

def symbolic_regression(train_df, initial_population):
    rng = np.random.default_rng()
    x_values = train_df['x'].tolist()
    y_true = train_df['f(x)'].tolist()
    population = []
    fitnesses = []
    print("initialize population fitness")
    # initialize fitness for initial population
    for individual in initial_population:
        individual.fitness = get_fitness(individual, x_values, y_true)
        fitnesses.append(individual.fitness)

    print("start genetic algorithm")
    population = initial_population
    generation_count = 0
    while generation_count < GENERATIONS and ((1 - max(fitnesses)) > 1e-6): 
        print(f"Generation {generation_count} started")
        new_population = []
        new_fitnesses = []
        for _ in range(INITIAL_POPULATION_SIZE):
            # normalize weights so they sum to 1.0
            weights = fitnesses / np.sum(fitnesses)
            p = np.random.random()
            if p < CROSSOVER_RATE:
                # select two parents 
                # (implement logic to prioritize higher fitness)

                parent1 = rng.choice(population, p=weights)
                parent2 = rng.choice(population, p=weights)

                # perform crossover
                child = crossover(parent1.copy(), parent2.copy())
            elif p < CROSSOVER_RATE + CLONE_RATE:
                # select one parent
                parent = rng.choice(population, p=weights)
                # clone the parent
                child = parent.copy()
            else:
                # select one parent
                parent = rng.choice(population, p=weights)
                # perform mutation
                child = mutate(parent.copy(), OPERATORS, X_PROBABILITY)
            child.fitness = get_fitness(child, x_values, y_true)
            new_population.append(child)
            new_fitnesses.append(child.fitness)
        population = new_population
        fitnesses = new_fitnesses
        generation_count += 1

    max_fitness_index = np.argmax(fitnesses)
    return population[max_fitness_index]

def get_fitness(individual, x_values, y_true):
    y_pred = []
    for x in x_values:
        y_pred.append(individual.evaluate(x))
    return 1/ (1 + root_mean_squared_error(y_true, y_pred))

def test(individual, x_values, y_true):
    y_pred = []
    for x in x_values:
        y_pred.append(individual.evaluate(x))
    return root_mean_squared_error(y_true, y_pred)

def main():
    df = pd.read_csv('data/dataset1.csv')

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
    x_true_values = test_df['x'].tolist()
    y_true_values = test_df['f(x)'].tolist()
    rmse = test(most_fit, x_true_values, y_true_values)
    print(f"fitness: {most_fit.fitness}")
    print(f"RMSE: {rmse}")
    
if __name__ == "__main__":
    main()