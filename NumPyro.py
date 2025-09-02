# bayesian_mlp_numpyro.py
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


X = np.linspace(-1, 1, 20)
Y = X**3 + 0.1 * np.random.randn(20)

def model(X, Y=None):
    w = numpyro.sample("w", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = w * X + b
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=Y)

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), X, Y)
mcmc.print_summary()
