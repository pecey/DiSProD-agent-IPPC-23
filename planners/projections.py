import jax
import jax.numpy as jnp


def project_dummy(nA):
    def _project_dummy(ac):
        return ac
    return _project_dummy

# Projection function for sum = 1 constraint 
# https://arxiv.org/pdf/1309.1541.pdf
def project_sum_one(nA):
    def _projection_fn(ac):
        ac_sort = jnp.sort(ac)[::-1]
        ac_sort_cumsum = jnp.cumsum(ac_sort)
        rho_candidates = ac_sort + (1 - ac_sort_cumsum)/jnp.arange(1, nA+1)
        mask = jnp.where(rho_candidates > 0, jnp.arange(nA, dtype=jnp.int32), -jnp.ones(nA, dtype=jnp.int32))
        rho = jnp.max(mask)
        contrib = (1 - ac_sort_cumsum[rho])/(rho + 1)
        return jax.nn.relu(ac + contrib)
    return _projection_fn

def project_recsim(nA, n_consumer, n_item):
    _row_projector = project_sum_one(n_item)
    row_projector_fn = jax.vmap(_row_projector, in_axes=(0), out_axes=(0))
    matrix_projector_fn = project_sum_one(nA)
    
    def _project_recsim(ac):
        # Constraint 1: Each row should sum upto 1.
        ac_shaped = ac.reshape(n_consumer, n_item)
        ac_shaped = row_projector_fn(ac_shaped)
        ac = ac_shaped.flatten()
        # Constraint 2: Entire matrix should sum upto 1
        ac = matrix_projector_fn(ac)
        return ac
    return _project_recsim



