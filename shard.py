import jax
import qax
import numpy as np
import jax.tree_util as jtu
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from aqt.jax.v2.aqt_dot_general import make_dot_general, make_conv_general_dilated
from aqt.jax.v2 import config as aqt_config

proj_ops = [jax.lax.dot_general_p, jax.lax.conv_general_dilated_p]


@dataclass
class QuantProjKernel(qax.ImplicitArray):
    val: jax.Array

    def materialize(self):
        return self.val


@qax.primitive_handler(proj_ops)
def quant_proj_handler(primitive, lhs, rhs: QuantProjKernel, **kwargs):
    cfg = aqt_config.fully_quantized(8, 8)
    if primitive is jax.lax.dot_general_p:
        val = make_dot_general(cfg)(lhs, rhs.val, **kwargs)
    else:
        val = make_conv_general_dilated(cfg.fwd)(lhs, rhs.val, **kwargs)
    return val


@dataclass
class LatentProjTracer(qax.ImplicitArray):
    val: jax.Array
    row_wise: bool = qax.aux_field()
    align_trace: dict = qax.aux_field()

    def log_kernel(self, args):
        rhs = next((arg for arg in args if isinstance(arg, KernelProjTracer)), None)
        if rhs is not None:
            self.align_trace[rhs.name] = self.row_wise

    def default_handler(self, primitive, *args, params=None):
        val = super().default_handler(primitive, *args, params=params)
        self.log_kernel(args)
        # if primitive in (jax.lax.dot_general_p, jax.lax.conv_general_dilated_p):
        #     self.log_kernel(args)
        #     transposition = not self.transposition
        # else:
        #     transposition = self.transposition
        # return LatentProjTracer(val, transposition, self.align_trace)
        return LatentProjTracer(val, self.row_wise, self.align_trace)

    def materialize(self):
        return self.val


@dataclass
class KernelProjTracer(qax.ImplicitArray):
    name: str = qax.aux_field()
    val: jax.Array

    def default_handler(self, primitive, *args, params=None):
        # lhs = next((arg for arg in args if isinstance(arg, LatentProjTracer)), None)
        # if lhs is not None:
        #     val = lhs.default_handler(primitive, *args, params=params)
        # else:
        val = super().default_handler(primitive, *args, params=params)
        return KernelProjTracer(self.name, val)

    def materialize(self):
        return self.val


# proj_ops = jax.lax.dot_general_p


@qax.primitive_handler(proj_ops)
def traced_proj_handler(
    primitive, lhs: LatentProjTracer, rhs: KernelProjTracer, **kwargs
):
    align_trace = lhs.align_trace
    align_trace[rhs.name] = lhs.row_wise
    new_row_wise = not lhs.row_wise

    if primitive is jax.lax.dot_general_p:
        val = jax.lax.dot_general(lhs, rhs, **kwargs)
    else:
        val = jax.lax.conv_general_dilated(lhs, rhs, **kwargs)
    return LatentProjTracer(val, new_row_wise, align_trace)


@dataclass
class LoudArray(qax.ImplicitArray):
    name: str = qax.aux_field()
    val: jax.Array

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        for i, arg in enumerate(args):
            if isinstance(arg, LoudArray):
                print(f"{arg.name} is used in {primitive} as arg {i}")
                print(f"\t params: {params}")
        return super().default_handler(primitive, *args, params=params)

    def materialize(self):
        return self.val


def flatten_and_shard(
    inner: optax.GradientTransformation,
) -> optax.GradientTransformationExtraArgs:
    """MOD(flenser): this is a modified version of optax.flatten that enforces a
    cross-device sharding constraint against the state vector and pads it to the
    required length to be sharded unit-wise.

    Flattens parameters and gradients for init and update of inner transform.

    This can reduce the overhead of performing many calculations on lots of small
    variables, at the cost of slightly increased memory usage.


    Args:
      inner: Inner transformation to flatten inputs for.

    Returns:
      New ``GradientTransformationExtraArgs``
    """

    inner = optax.with_extra_args_support(inner)

    def _flat_len(updates):
        return sum(a.size for a in jtu.tree_leaves(updates))

    def _pad_len(updates):
        updates, _ = jtu.tree_flatten(updates)
        dev_num = jax.local_device_count()
        return dev_num - _flat_len(updates) % dev_num

    def _flatten(params):
        """Flattens and concatenates all tensors in params to a single vector."""
        params, _ = jtu.tree_flatten(params)
        vector = jnp.concatenate([jnp.reshape(param, [-1]) for param in params])
        vector = jnp.pad(vector, (0, _pad_len(params)))
        vector = jax.lax.with_sharding_constraint(vector, shards)
        return vector

    def _unflatten(updates, flat):
        """Extracts tensors from flat, using the structure and shapes of params."""
        updates_flat, treedef = jtu.tree_flatten(updates)
        offsets = []
        for update in updates_flat:
            size = np.prod(update.shape)
            if offsets:
                offsets.append(size + offsets[-1])
            else:
                offsets.append(size)
        del offsets[-1]
        flat = flat[: flat.size - _pad_len(updates)]
        flat_split = jnp.split(flat, offsets)
        reshaped = [
            jnp.reshape(flat_update, update.shape)
            for flat_update, update in zip(flat_split, updates_flat)
        ]
        return jtu.tree_unflatten(treedef, reshaped)

    shards = jax.sharding.PositionalSharding(jax.local_devices())

    def init_fn(params):
        return inner.init(_flatten(params))

    def update_fn(updates, state, params=None, **extra_args):
        if params is not None:
            params = _flatten(params)
        updates_flat = _flatten(updates)
        updates_flat, state = inner.update(
            _flatten(updates), state, params, **extra_args
        )
        updates = _unflatten(updates, updates_flat)
        return updates, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
