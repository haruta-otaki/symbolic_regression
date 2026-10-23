import random
import time

import sympy as sp
import pandas as pd
import numpy as np
from Tree import Node, generate_random_tree, mutate, crossover

INITIAL_POPULATION_SIZE = 500
MAX_DEPTH = 5
MAX_TREE_DEPTH = 10          # cap after crossover / mutation (bloat control)
BINARY_OPERATORS = ["+", "-", "*", "/"]
UNARY_OPERATORS = ["sin", "cos", "log", "exp"]

GENERATIONS = 50
CROSSOVER_RATE = 0.7
CLONE_RATE = 0.1
MUTATION_RATE = 0.2          # mutation = whatever is left after crossover and clone

X_PROBABILITY = 0.5
LEAF_PROBABILITY = 0.2
UNARY_PROBABILITY = 0.2
TOURNAMENT_SIZE = 5
ELITE_RATIO = 0.01
PENALTY_WEIGHT = 0.01        # complexity penalty per node
SPLIT_X = -2.0               # dataset 3: boundary between the huge-value and small-value regions

DATASET = 3

# set by initialize()
VARIABLES = ["x"]
TARGET = "f(x)"
REAL_CONSTANTS = False
P_UNARY = 0.0


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def initialize(dataset=None):
    """Sets the dataset-specific settings, loads the CSV, and returns
    X_train, y_train, X_test, y_test (plus the test DataFrame for reports)."""
    global DATASET, VARIABLES, TARGET, REAL_CONSTANTS, P_UNARY
    if dataset is not None:
        DATASET = dataset

    if DATASET == 2:
        VARIABLES = ["x1", "x2", "x3"]
        TARGET = "f(x1,x2,x3)"
        REAL_CONSTANTS = True
        P_UNARY = 0.0
        df = pd.read_csv("data/dataset2.csv", skiprows=1, names=VARIABLES + [TARGET])
        before = len(df)
        df = df[df["x3"] >= 0.1]
        print(f"removed {before - len(df)} rows with x3 < 0.1 (too imprecise after rounding)")
    else:
        VARIABLES = ["x"]
        TARGET = "f(x)"
        REAL_CONSTANTS = False
        P_UNARY = UNARY_PROBABILITY if DATASET == 3 else 0.0
        df = pd.read_csv(f"data/dataset{DATASET}.csv")

    # select 80% of the data randomly for training, the rest for testing
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    X_train, y_train = to_arrays(train_df, VARIABLES, TARGET)
    X_test, y_test = to_arrays(test_df, VARIABLES, TARGET)
    return X_train, y_train, X_test, y_test, test_df


def to_arrays(df, x, y):
    X = {}
    for v in x:
        X[v] = df[v].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    return X, Y


def generate_initial_population():
    return [generate_random_tree(BINARY_OPERATORS, UNARY_OPERATORS, P_UNARY, MAX_DEPTH,
                                 LEAF_PROBABILITY, X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
            for _ in range(INITIAL_POPULATION_SIZE)]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def tournament_selection(population, fitnesses, rng):
    indices = rng.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    winner = indices[np.argmax(np.asarray(fitnesses)[indices])]
    return population[winner]


def roulette_selection(population, fitnesses, rng):
    f = np.asarray(fitnesses, dtype=float)
    total = f.sum()
    if not np.isfinite(total) or total <= 0:
        return population[rng.integers(len(population))]      # every tree invalid: pick at random
    return population[rng.choice(len(population), p=f / total)]


# ---------------------------------------------------------------------------
# genetic algorithm
# ---------------------------------------------------------------------------
def symbolic_regression(x_values, y_true, tournament=True, elitism=True, penalty=True,
                        seed=None, verbose=True):
    """Runs the GP and returns (best individual, history).
    history has one dict per generation, for charts and convergence statistics.
      tournament: True = tournament selection, False = roulette (fitness-proportional)
      elitism:    False = no elites are copied (elite count 0)
      penalty:    False = no complexity penalty in the fitness
      seed:       set it to make the run repeatable"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)
    select = tournament_selection if tournament else roulette_selection
    elitist_population = int(round(INITIAL_POPULATION_SIZE * ELITE_RATIO)) if elitism else 0

    population = generate_initial_population()
    for individual in population:
        individual.error, individual.fitness = get_evaluation(individual, x_values, y_true, penalty)
    best_ever = min(population, key=lambda t: t.error).copy()

    history = []
    start = time.time()
    for generation in range(GENERATIONS):
        fitnesses = [t.fitness for t in population]

        # elitism: copy the best trees unchanged
        order = np.argsort(fitnesses)[::-1]
        new_population = [population[i].copy() for i in order[:elitist_population]]

        while len(new_population) < INITIAL_POPULATION_SIZE:
            p = rng.random()
            if p < CROSSOVER_RATE:
                parent1 = select(population, fitnesses, rng)
                parent2 = select(population, fitnesses, rng)
                child = crossover(parent1, parent2)
                if child.depth() > MAX_TREE_DEPTH:          # reject bloated children
                    child = parent1.copy()
            elif p < CROSSOVER_RATE + CLONE_RATE:
                child = select(population, fitnesses, rng).copy()
            else:
                parent = select(population, fitnesses, rng)
                child = mutate(parent, BINARY_OPERATORS, UNARY_OPERATORS, P_UNARY,
                               X_PROBABILITY, VARIABLES, REAL_CONSTANTS)
                if child.depth() > MAX_TREE_DEPTH:
                    child = parent.copy()
            child.error, child.fitness = get_evaluation(child, x_values, y_true, penalty)
            new_population.append(child)
        population = new_population

        # statistics for this generation
        errors = np.array([t.error for t in population])
        finite = errors[np.isfinite(errors)]
        best = population[int(np.argmin(errors))]
        if best.error < best_ever.error:
            best_ever = best.copy()
        history.append(dict(
            generation=generation,
            best_error=best.error,                          # best in this generation
            best_ever_error=best_ever.error,                # best found so far
            median_error=np.median(finite) if len(finite) else np.inf,
            mean_size=np.mean([t.size() for t in population]),
            unique_fraction=len({t.to_infix() for t in population}) / INITIAL_POPULATION_SIZE,
            seconds=time.time() - start,
        ))
        if verbose:
            print(f"Generation {generation}: Current Error = {best.error:.6f}")
        if best_ever.error <= 1e-6:
            break

    return best_ever, history


# ---------------------------------------------------------------------------
# fitness
# ---------------------------------------------------------------------------
def get_evaluation(individual, X, y, penalty=True):
    """Returns (error, fitness). penalty=False leaves out the complexity penalty."""
    weight = PENALTY_WEIGHT if penalty else 0.0
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
        with np.errstate(all="ignore"):
            error = np.sqrt(np.mean((y - (individual.shift + coefficient * y_pred)) ** 2))
    elif DATASET == 2:
        # f is rounded to 2 decimals, so a printed 0.00 is between 0 and 0.005
        y = np.maximum(y, 0.001)
        # RMSLE needs positive predictions
        if np.any(y_pred <= 0):
            return (float("inf"), 0.0)
        log_gap = np.log(y) - np.log(y_pred)      # log of (true / predicted) per row
        log_c = np.mean(log_gap)                  # best multiplier, in log form
        individual.shift = 0.0
        individual.coefficient = np.exp(log_c)
        error = np.sqrt(np.mean((log_gap - log_c) ** 2))
    else:
        # f is huge for x < -2 (thousands) and small for x >= -2 (below ~10), so plain RMSE
        # would only care about the huge values. Split the rows into those two regions and
        # make each region count equally: every row is weighted by 1 / (rows in its region
        # * variance of y in its region), i.e. each region's error is judged against its own scale.
        w = np.zeros_like(y)
        for region in (X["x"] < SPLIT_X, X["x"] >= SPLIT_X):
            if region.sum() > 1:
                w[region] = 1 / (region.sum() * np.var(y[region]))
        w = w / w.sum()                                   # weights sum to 1
        wm = lambda v: np.sum(w * v)                      # weighted average

        # weighted linear scaling: best a + b * prediction under the same weights,
        # so the constants also care about both regions
        var_w = wm((y_pred - wm(y_pred)) ** 2)
        if not np.isfinite(var_w) or var_w < 1e-12:      # constant tree (e.g. "0"): reject
            return (float("inf"), 0.0)
        coefficient = wm((y_pred - wm(y_pred)) * (y - wm(y))) / var_w
        individual.coefficient = coefficient
        individual.shift = wm(y) - coefficient * wm(y_pred)

        # balanced error: root mean of the two regions' squared NRMSE
        # (each region's RMSE divided by that region's spread of y; 0 = perfect)
        with np.errstate(all="ignore"):
            error = np.sqrt(2 * wm((y - (individual.shift + coefficient * y_pred)) ** 2))

    if not np.isfinite(error):
        return (float("inf"), 0.0)
    # complexity penalty
    return (error, 1 / (1 + error + weight * individual.size()))


def get_error(individual, X, y, df_copy):
    with np.errstate(all="ignore"):
        y_pred = individual.coefficient * individual.evaluate(X) + individual.shift
    # check if all predictions are finite
    if not np.all(np.isfinite(y_pred)):
        return (float("inf"), 0.0)

    if DATASET == 2:
        df_copy["pred"] = y_pred
        df_copy["abs_error"] = np.abs(df_copy[TARGET] - df_copy["pred"])
        # sort by error, biggest first, and show the top 20
        print(df_copy.sort_values("abs_error", ascending=False).head(20))

    rel = np.abs(y - y_pred) / np.maximum(np.abs(y), 0.005)
    print(f"Median relative error: {100 * np.median(rel):.3f}%")
    # calculate rmse
    error = np.sqrt(np.mean((y - y_pred) ** 2))
    return (error, 1 / (1 + error * (1 + PENALTY_WEIGHT * individual.size())))


def main():
    X_train, y_train, X_test, y_test, test_df = initialize(DATASET)
    print("Dataset:", DATASET)

    print("start symbolic regression")
    most_fit, history = symbolic_regression(X_train, y_train)
    most_fit_str = most_fit.to_infix()
    print(f"Original Individual: {most_fit_str}")
    expr = sp.sympify(most_fit_str)
    print("Simplified:", sp.simplify(most_fit.coefficient * expr + most_fit.shift))

    # testing:
    error, fitness = get_error(most_fit, X_test, y_test, test_df.copy())
    print(f"Test Coef: {most_fit.coefficient}")
    print(f"Test Error: {error}")
    print(f"Test Fitness: {fitness}")


if __name__ == "__main__":
    main()