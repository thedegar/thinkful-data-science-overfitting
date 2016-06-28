#####################################################
# Tyler Hedegard
# 6/28/2016
# Thinkful Data Science
# Overfitting
#####################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()
# R-squared = 0.642, p = 0.000
poly_1_test = smf.ols(formula='y ~ 1 + X', data=test_df).fit()
# R-squared = 0.958, p = 0.000

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()
# R-squared = 0.666, p = 0.000
poly_2_test = smf.ols(formula='y ~ 1 + X + I(X**2)', data=test_df).fit()
# R-squared = 0.964, p = 0.000

""" Conclusion:
The quadratic and linear fit models produce very similar results.
Therefore using the more simple Linear Fit is appropriate.
"""