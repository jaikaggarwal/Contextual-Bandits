import pandas as pd
import numpy.random as nprnd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy.stats import invgamma
from scipy.stats import spearmanr

# First let us set up our sample size variable to determine total nnumber of users
sample_size = 100

# We also need number of charities
num_charities = 3


# Now we can create our contextual variable that holds binary values (political affiliation)
republicans = nprnd.randint(2, size=sample_size)

# Now we want our experimental variables
matching = nprnd.randint(2, size=sample_size)

# Create charity data
charities = nprnd.randint(num_charities, size=sample_size)

# Set charities as binary variables

charity_2 = charities == 1
charity_3 = charities == 2

print(charities, charity_2, charity_3)

df = pd.DataFrame(
    {'Republicans': republicans,
     'Matching': matching,
     'Charity2': charity_2,
     "Charity3": charity_3
     })

# Given our true model, we can generate our data (DGP)
df["donation"] = (40 + 10*df["Charity2"] + 20*df["Charity3"]
                  + 15*df["Matching"] - 10*df["Republicans"] -5*df["Matching"]*df["Republicans"]
                  + 5*df["Charity2"]*df["Matching"] - 5*df["Charity3"]*df["Matching"]
                  + nprnd.normal(scale=1, size=sample_size))
x = nprnd.normal(scale=1, size=sample_size)

regression = sm.ols()
