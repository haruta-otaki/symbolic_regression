
import pandas as pd
import math
import random
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

DATASET = 1

# Number of rows used for fitness evaluation
# during each generation.
SAMPLE_SIZE = 1000

# Refresh the training sample every 5 generations.
SAMPLE_REFRESH = 5

TOURNAMENT_SIZE = 5

if DATASET == 1:
    VARIABLES = ["x"]
    REAL_CONSTANTS = False
else:
    VARIABLES = ["x1", "x2", "x3"]
    REAL_CONSTANTS = True


def generate_initial_population():
    initial_population = []

    for _ in range(INITIAL_POPULATION_SIZE):
        individual = generate_random_tree(
            OPERATORS,
            MAX_DEPTH,
            LEAF_PROBABILITY,
            X_PROBABILITY,
            VARIABLES,
            REAL_CONSTANTS
        )

        initial_population.append(individual)

    return initial_population


def get_data_values(df):
    if DATASET == 1:
        x_values = df["x"].tolist()
        y_true = df["f(x)"].tolist()
    else:
        x_values = df[["x1", "x2", "x3"]].to_dict("records")
        y_true = df["f(x1,x2,x3)"].tolist()

    return x_values, y_true


def tournament_selection(population, tournament_size=5):
    competitors = random.sample(population, tournament_size)

    return max(
        competitors,
        key=lambda individual: individual.fitness
    )


def get_predictions(individual, x_values):
    predictions = []

    try:
        for x in x_values:
            prediction = individual.evaluate(x)

            if not math.isfinite(prediction):
                return None

            predictions.append(prediction)

        return np.array(predictions, dtype=float)

    except (
        ValueError,
        OverflowError,
        ZeroDivisionError,
        FloatingPointError
    ):
        return None


def get_fitness(individual, x_values, y_true):
    y_pred = get_predictions(individual, x_values)

    if y_pred is None:
        return 0.0

    y_true = np.asarray(y_true, dtype=float)

    try:
        if DATASET == 2:
            # Log error works better when output values
            # span several orders of magnitude.
            if np.any(y_pred <= 0):
                return 0.0

            with np.errstate(over="raise", invalid="raise"):
                error = np.sqrt(
                    np.mean(
                        (
                            np.log1p(y_true) -
                            np.log1p(y_pred)
                        ) ** 2
                    )
                )
        else:
            # Keep original RMSE fitness for Dataset 1.
            error = root_mean_squared_error(y_true, y_pred)

        if not math.isfinite(error):
            return 0.0

        # Soft complexity penalty only.
        # This does not limit tree size or depth.
        complexity_penalty = 0.001 * individual.size()

        fitness = 1 / (1 + error + complexity_penalty)

        return float(fitness)

    except (
        ValueError,
        OverflowError,
        ZeroDivisionError,
        FloatingPointError
    ):
        return 0.0


def get_rmse(individual, x_values, y_true):
    y_pred = get_predictions(individual, x_values)

    if y_pred is None:
        return float("inf")

    try:
        error = root_mean_squared_error(y_true, y_pred)

        if not math.isfinite(error):
            return float("inf")

        return float(error)

    except (
        ValueError,
        OverflowError,
        ZeroDivisionError,
        FloatingPointError
    ):
        return float("inf")


def symbolic_regression(train_df, initial_population):
    population = initial_population

    # Keep the best equation from each generation.
    # Later we compare these on the full training set.
    champions = []

    for generation in range(GENERATIONS):

        # Refresh the sample every 5 generations.
        if generation % SAMPLE_REFRESH == 0:
            sample_df = train_df.sample(
                n=min(SAMPLE_SIZE, len(train_df)),
                random_state=42 + generation
            )

            x_values, y_true = get_data_values(sample_df)

            # Recalculate all fitness values because
            # the training sample has changed.
            for individual in population:
                individual.fitness = get_fitness(
                    individual, x_values, y_true
                )

            print(
                f"\nGeneration {generation}: "
                f"New training sample ({len(sample_df)} rows)"
            )

        # Elitism: preserve the best individual.
        best_individual = max(
            population,
            key=lambda individual: individual.fitness
        ).copy()

        new_population = [best_individual]

        for _ in range(INITIAL_POPULATION_SIZE - 1):
            p = random.random()

            if p < CROSSOVER_RATE:
                parent1 = tournament_selection(
                    population, TOURNAMENT_SIZE
                )
                parent2 = tournament_selection(
                    population, TOURNAMENT_SIZE
                )

                child = crossover(parent1, parent2)

            elif p < CROSSOVER_RATE + CLONE_RATE:
                parent = tournament_selection(
                    population, TOURNAMENT_SIZE
                )

                child = parent.copy()

            else:
                parent = tournament_selection(
                    population, TOURNAMENT_SIZE
                )

                child = mutate(
                    parent,
                    OPERATORS,
                    X_PROBABILITY,
                    VARIABLES,
                    REAL_CONSTANTS
                )

            child.fitness = get_fitness(
                child, x_values, y_true
            )

            new_population.append(child)

        population = new_population

        best_individual = max(
            population,
            key=lambda individual: individual.fitness
        )

        champions.append(best_individual.copy())

        print(
            f"Generation {generation}: "
            f"Best Fitness = {best_individual.fitness:.6f}, "
            f"Size = {best_individual.size()}, "
            f"Depth = {best_individual.depth()}"
        )

    # Compare generation winners on the full
    # training set, not on different small samples.
    print("\nEvaluating champions on full training data...")

    full_x, full_y = get_data_values(train_df)

    for individual in champions:
        individual.fitness = get_fitness(
            individual, full_x, full_y
        )

    most_fit = max(
        champions,
        key=lambda individual: individual.fitness
    )

    return most_fit


def main():
    if DATASET == 2:
        # Dataset 2 has an unquoted header containing commas.
        df = pd.read_csv(
            "data/dataset2.csv",
            skiprows=1,
            names=["x1", "x2", "x3", "f(x1,x2,x3)"]
        )
    else:
        df = pd.read_csv("data/dataset1.csv")

    print("Dataset:", DATASET)
    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))

    # 80% training, 20% testing.
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    print("\nGenerating initial population...")
    initial_population = generate_initial_population()

    print("Starting symbolic regression...")
    most_fit = symbolic_regression(
        train_df, initial_population
    )

    print("\n========== BEST EQUATION ==========")
    print(most_fit.to_infix())

    print("\n========== TREE INFORMATION ==========")
    print("Tree Size:", most_fit.size())
    print("Tree Depth:", most_fit.depth())

    # The tree drawing may be extremely wide
    # when crossover produces a large tree.
    print("\n========== TREE ==========")
    print(most_fit)

    # Evaluate on all training observations.
    train_x, train_y = get_data_values(train_df)

    train_fitness = get_fitness(
        most_fit, train_x, train_y
    )

    train_rmse = get_rmse(
        most_fit, train_x, train_y
    )

    # Evaluate on untouched test observations.
    test_x, test_y = get_data_values(test_df)

    test_fitness = get_fitness(
        most_fit, test_x, test_y
    )

    test_rmse = get_rmse(
        most_fit, test_x, test_y
    )

    print("\n========== FINAL RESULTS ==========")
    print(f"Training Fitness: {train_fitness:.6f}")
    print(f"Training RMSE: {train_rmse:.6f}")

    print(f"Test Fitness: {test_fitness:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    if DATASET == 2:
        print(
            "\nNote: Dataset 2 fitness uses log error "
            "plus the complexity penalty."
        )
        print("RMSE is calculated on the original output scale.")


if __name__ == "__main__":
    main()
