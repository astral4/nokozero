"""The Q-network: a transformer over entity tokens with a dueling Bernoulli head.

The output is a logit per action. Its sigmoid is the estimated probability of reaching the segment's end from the state
under that action. Dueling is applied in logit space (a state logit plus mean-centered per-action advantages).
"""

# JAX, Equinox, and Optax only come with partial annotations.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false, reportMissingSuperCall=false

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, Bool, Float  # noqa: TC002

# ruff: noqa: F722  # jaxtyping shape strings are not forward references


class Block(eqx.Module):
    """Pre-norm transformer block with masked self-attention."""

    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP

    def __init__(self, dim: int, heads: int, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.attn = eqx.nn.MultiheadAttention(heads, dim, key=k1)
        self.mlp = eqx.nn.MLP(dim, dim, 2 * dim, depth=1, activation=jax.nn.gelu, key=k2)

    def __call__(self, x: Float[Array, "k d"], mask: Bool[Array, "k k"]) -> Float[Array, "k d"]:
        """Apply attention and the MLP with residuals; `mask[q, k]` allows query q to see key k."""
        h = jax.vmap(self.ln1)(x)
        x = x + self.attn(h, h, h, mask=mask)
        return x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))


class QNet(eqx.Module):
    """Token encoder, attention pooling, global features, dueling head."""

    embed: eqx.nn.MLP
    null_token: jax.Array
    blocks: list[Block]
    pool_query: jax.Array
    pool: eqx.nn.MultiheadAttention
    ln_pool: eqx.nn.LayerNorm
    glob: eqx.nn.MLP
    trunk: eqx.nn.MLP
    value: eqx.nn.Linear
    advantage: eqx.nn.Linear
    token_dim: int = eqx.field(static=True)
    global_dim: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        token_dim: int,
        global_dim: int,
        num_actions: int,
        *,
        dim: int = 128,
        heads: int = 4,
        depth: int = 2,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 10 + depth)
        self.embed = eqx.nn.MLP(token_dim, dim, dim, depth=1, activation=jax.nn.gelu, key=keys[0])
        self.null_token = jax.random.normal(keys[1], (1, dim)) * 0.02
        self.blocks = [Block(dim, heads, keys[10 + i]) for i in range(depth)]
        self.pool_query = jax.random.normal(keys[2], (1, dim)) * 0.02
        self.pool = eqx.nn.MultiheadAttention(heads, dim, key=keys[3])
        self.ln_pool = eqx.nn.LayerNorm(dim)
        self.glob = eqx.nn.MLP(global_dim, dim, dim, depth=1, activation=jax.nn.gelu, key=keys[4])
        self.trunk = eqx.nn.MLP(2 * dim, dim, dim, depth=1, activation=jax.nn.gelu, key=keys[5])
        self.value = eqx.nn.Linear(dim, 1, key=keys[6])
        self.advantage = eqx.nn.Linear(dim, num_actions, key=keys[7])
        self.token_dim, self.global_dim, self.num_actions = token_dim, global_dim, num_actions
        self.dim, self.heads, self.depth = dim, heads, depth

    @property
    def hyperparameters(self) -> dict[str, int]:
        """Arguments for rebuilding the network via `QNet(**hyperparameters, key=...)`."""
        return {
            "token_dim": self.token_dim,
            "global_dim": self.global_dim,
            "num_actions": self.num_actions,
            "dim": self.dim,
            "heads": self.heads,
            "depth": self.depth,
        }

    def __call__(
        self, tokens: Float[Array, "k f"], mask: Bool[Array, " k"], glob: Float[Array, " g"]
    ) -> Float[Array, " a"]:
        """Return one logit per action for a single state."""
        x = jnp.concatenate([self.null_token, jax.vmap(self.embed)(tokens)], axis=0)
        m = jnp.concatenate([jnp.ones((1,), dtype=bool), mask])
        attn_mask = jnp.broadcast_to(m[None, :], (m.shape[0], m.shape[0]))
        for block in self.blocks:
            x = block(x, attn_mask)
        x = jax.vmap(self.ln_pool)(x)
        pooled = self.pool(self.pool_query, x, x, mask=m[None, :])[0]
        h = self.trunk(jnp.concatenate([pooled, self.glob(glob)]))
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean()
