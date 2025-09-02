# gradient_example_jax.py
import jax
import jax.numpy as jnp


def f(x):
    return jnp.sin(x) * jnp.exp(-0.1*x)


df = jax.grad(f)

x_val = 2.0
print("f(x) =", f(x_val))
print("df/dx =", df(x_val))
