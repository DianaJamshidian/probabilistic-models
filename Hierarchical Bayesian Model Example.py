# hierarchical_bayesian_model.py
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
n_schools = 5
students_per_school = 30
true_mu_school = [70, 75, 80, 65, 90]  # میانگین نمرات واقعی هر مدرسه
sigma_school = 5


scores = [np.random.normal(mu, sigma_school, students_per_school) for mu in true_mu_school]


with pm.Model() as model:

    mu_overall = pm.Normal("mu_overall", mu=75, sigma=10)
    sigma_overall = pm.HalfNormal("sigma_overall", sigma=10)
    

    mu_school = pm.Normal("mu_school", mu=mu_overall, sigma=sigma_overall, shape=n_schools)
    

    for i in range(n_schools):
        pm.Normal(f"scores_{i}", mu=mu_school[i], sigma=sigma_school, observed=scores[i])
    

    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)


az.plot_trace(trace)
plt.show()


summary = az.summary(trace, hdi_prob=0.95)
print(summary)
