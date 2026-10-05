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

import pandas as pd
import numpy as np

df = pd.read_csv('data/dataset1.csv')

# select 80% of the data randomly for training
train_df = df.sample(frac=0.8, random_state=42)

# select the remaining 20% for testing using dropped indices
test_df = df.drop(train_df.index)

# generate 



def symbolic_regression():
    # loop 

    return # function

#evaluate



