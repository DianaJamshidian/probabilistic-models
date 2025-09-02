# bayesian_linear_regression.py
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.linspace(0, 10, 50)
true_slope = 2.5
true_intercept = 1.0
Y = true_slope * X + true_intercept + np.random.normal(0, 1, size=50)


with pm.Model() as model:
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    mu = slope * X + intercept
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)
    
    trace = pm.sample(1000, tune=1000, cores=1)

pm.plot_trace(trace)
plt.show()
