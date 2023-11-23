from dataclasses import replace
import fire
import qax
from diffusers import FlaxDDPMScheduler
import warnings
from functools import partial
import numpy as np
import flax.traverse_util as ftu
from transformers import CLIPTokenizer
import rich.progress as rp
import itertools as it
import jax
import optax
import lorax
import jax.numpy as jnp
import polars as pl
import os.path as osp
from flax.training import train_state
import jax.tree_util as jtu
from istrm import fetch_posts_meta
from train_utils import (
    input_stream,
    drop_posemb,
    objective,
    Inputs,
    Pipeline,
)
from shard import LatentProjTracer, KernelProjTracer, QuantProjKernel


class TrainState(train_state.TrainState):
    target: optax.Params
    rng: jax.random.PRNGKey
    loss: jax.Array
    loss_avg: optax.EmaState


def fetch_posts():
    posts = fetch_posts_meta()
    posts = (
        posts.lazy()
        .join(
            pl.scan_csv("posts.csv").select("id", "fav_count"),
            left_on="post_id",
            right_on="id",
        )
        .join(pl.scan_csv("score.csv"), left_on="post_id", right_on="id")
        .with_columns(
            (
                (pl.col("fav_count") - pl.col("fav_count").mean())
                / pl.col("fav_count").std()
                - pl.col("fav_score")
            ).alias("score"),
            pl.col("caption").str.split(", ").list.slice(1),
        )
        .collect()
    )
    return posts


def emplace(flat_params, flat_traces):
    tp = jax.sharding.PositionalSharding(jax.local_devices()).reshape(1, -1)
    conv_tp = tp.reshape(1, 1, 1, -1)
    sharding = {}

    def sh(path, kernel, col_wise):
        if path.endswith("embedding"):
            return tp.replicate()
        shape = np.array(kernel.shape)
        if kernel.ndim == 4 and np.all(
            np.array(kernel.shape) > jax.local_device_count()
        ):
            return conv_tp if col_wise else np.moveaxis(conv_tp, -1, -2)
        if kernel.ndim == 2 and np.all((shape % np.array(tp.shape)) == 0):
            return tp if col_wise else tp.T
        elif (
            kernel.ndim == 1
            and kernel.size % jax.local_device_count() == 0
            and col_wise
        ):
            return tp.reshape(-1)
        else:
            return tp.reshape(-1).replicate()

    for k, v in flat_params.items():
        if k in flat_traces:
            row_wise = flat_traces[k]
            sharding[k] = sh(k, v, row_wise)
        else:
            sharding[k] = tp.reshape(-1).replicate()

    return sharding


def main(
    model_slug="SimianLuo/LCM_Dreamshaper_v7",
    posts_path="posts.parquet",
    batch_size=8,
    peak_lr=2e-3,
    base_b1=0.85,
    peak_b1=0.95,
    skip_steps=10,
    cg_dropout=0.10,
    train_steps=1024,
    adapter_rank=64,
    relora_every=384,
    init_scale: float = 0.01,
    gamma=0.1,
    seed=42,
):
    assert skip_steps > 0, "skip_steps must be positive"
    tokenizer = CLIPTokenizer.from_pretrained(model_slug, subfolder="tokenizer")
    if not osp.exists(posts_path):
        posts = fetch_posts()
        posts.write_parquet(posts_path)
    else:
        posts = pl.read_parquet(posts_path)

    data = iter(input_stream(posts, tokenizer, batch_size, cg_dropout))
    inputs = next(data)
    scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        num_train_timesteps=1280,
    )
    scheduler.config.rescale_betas_zero_snr = True

    with jax.default_device(jax.devices("cpu")[0]):
        pipe, params = Pipeline.init(
            model_slug, scheduler=scheduler, skip_steps=skip_steps
        )
    traces = {}
    traced_inputs = jtu.tree_map(lambda v: LatentProjTracer(v, False, traces), inputs)

    def tracer(p, v):
        if p[-1] != "embedding":
            name = "/".join(p)
            if "conv" in name:
                return KernelProjTracer(name, v)
            else:
                return KernelProjTracer(name, v)
        else:
            return LatentProjTracer(v, False, traces)

    traced_params = ftu.path_aware_map(tracer, params)

    def trainable(params):
        fully_replicated = jax.sharding.PositionalSharding(
            jax.local_devices()
        ).replicate()
        return {
            k: jax.lax.with_sharding_constraint(v.b, fully_replicated)
            for k, v in ftu.flatten_dict(params).items()
            if isinstance(v, lorax.LoraWeight)
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jax.eval_shape(
            qax.use_implicit_args(partial(objective, pipe=pipe)),
            online=traced_params,
            target=traced_params,
            frozen={},
            rng=jax.random.PRNGKey(0),
            inputs=traced_inputs,
        )
    shards = emplace(ftu.flatten_dict(params, sep="/"), traces)
    shards = ftu.unflatten_dict(shards, sep="/")
    params = drop_posemb(params)
    params = jax.device_put(params, shards)
    num_chunks, chunk_rem = divmod(train_steps, relora_every)
    chunk_lens = [relora_every] * num_chunks
    if chunk_rem != 0:
        chunk_lens.append(chunk_rem)
    chunks = [
        optax.linear_onecycle_schedule(
            chunk_len,
            peak_lr,
            pct_start=0.125,
            pct_final=0.875,
        )
        for chunk_len in chunk_lens
    ]
    lsched = optax.join_schedules(chunks, np.cumsum(chunk_lens[:-1]))
    # lsched = optax.linear_onecycle_schedule(train_steps, peak_lr)
    msched = lambda step: peak_b1 - (peak_b1 - base_b1) * (lsched(step) / peak_lr)
    from lion_quant import scale_by_lion_8bit

    gradtx = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.scale_by_lion)(msched),
        optax.inject_hyperparams(optax.scale)(lsched),
        optax.scale(-1),
    )

    # TODO: Figure out why flattening/sharding the optimizer states breaks everything
    # gradtx = optax.flatten(gradtx)
    # TODO: lookahead?
    # gradtx = optax.lookahead(gradtx, 8, 0.5)

    # TODO: figure out a way to compose 8-bit fwd/bwd with LoRa
    # def kernel(p, v):
    #     if p[-1] == jtu.DictKey("kernel"):
    #         return QuantProjKernel(v)
    #     else:
    #         return v

    def lora_rank(p, v):
        if p[:2] == ("vae", "encoder"):
            return lorax.LORA_FREEZE
        elif v.ndim > 1:
            return min(v.shape[-1], adapter_rank)
        elif v.ndim <= 1:
            return lorax.LORA_FULL
        else:
            return lorax.LORA_FREEZE

    lora_spec = ftu.path_aware_map(lora_rank, params)
    # params = qax.utils.tree_map_with_path_with_implicit(kernel, params)

    @jax.jit
    def train_init(params):
        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)

        params = jtu.tree_map(lambda v: v.astype(jnp.bfloat16), params)
        params = lorax.init_lora(
            params,
            rng=init_key,
            spec=lora_spec,
            stddev=init_scale,
        )
        return TrainState(
            rng=rng,
            step=jnp.zeros([], dtype=int),
            loss=jnp.zeros([]),
            loss_avg=optax.ema(0.99).init(jnp.zeros([])),
            apply_fn=lambda: None,
            params=params,
            target=params,
            tx=gradtx,
            opt_state=gradtx.init(trainable(params)),
        )

    tstate = jax.jit(train_init)(params)

    def re_lora(tstate: TrainState):
        leaves = qax.utils.tree_leaves_with_implicit(params)
        loras = [v for v in leaves if isinstance(v, lorax.LoraWeight)]
        rng, *keys = jax.random.split(tstate.rng, len(loras) + 1)
        for v, key in zip(loras, keys):
            v.w = v.materialize()
            v.a = jax.random.normal(key, v.a.shape, dtype=v.a.dtype) * init_scale
            v.b = jnp.zeros_like(v.b)

        # def reinit(v):
        #     if isinstance(v, lorax.LoraWeight):
        #         nonlocal rng
        #         rng, key = jax.random.split(rng)
        #         v.w += v.materialize()
        #         v.a = jax.random.normal(key, v.a.shape, dtype=v.a.dtype) * init_scale
        #         v.b = jnp.zeros_like(v.b)
        #     return v

        # params = qax.utils.tree_map_with_implicit(reinit, params)
        # return tstate.replace(params=params, rng=rng)
        return tstate.replace(rng=rng)

    @jax.value_and_grad
    # @jax.checkpoint
    def loss_fn(Bs, online, target, rng, inputs):
        return qax.use_implicit_args(partial(objective, pipe=pipe, gamma=gamma))(
            online, target, {}, rng=rng, inputs=inputs
        )

    def train_step(tstate: TrainState, inputs: Inputs):
        online, target = tstate.params, tstate.target
        rng, key = jax.random.split(tstate.rng)
        loss, grads = loss_fn(trainable(online), online, target, key, inputs)
        loss, loss_avg = optax.ema(0.99).update(loss, tstate.loss_avg)
        target = online
        updates, opt_state = tstate.tx.update(
            grads, tstate.opt_state, trainable(online)
        )
        updated = optax.apply_updates(trainable(online), updates)
        online = {
            k: replace(v, b=updated[k]) if k in updated else v
            for k, v in ftu.flatten_dict(online).items()
        }
        online = ftu.unflatten_dict(online)
        tstate = tstate.replace(
            params=online,
            target=target,
            opt_state=opt_state,
            loss=loss,
            loss_avg=loss_avg,
            rng=rng,
        )
        reinit = tstate.step % relora_every == 0
        tstate = jax.lax.cond(reinit, re_lora, lambda x: x, tstate)
        return tstate

    # TODO: implement a proper checkpointing system
    # TODO: implement a proper logging system (neptune.ai?)
    # TODO: implement a proper evaluation system

    with rp.Progress(
        rp.MofNCompleteColumn(),
        "{task.description} loss: {task.fields[loss]:.3g}",
        *rp.Progress.get_default_columns()[1:-1],
        rp.TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "compiling...",
            total=train_steps,
            loss=float("nan"),
            start=False,
        )
        p_train_step = jax.jit(
            train_step,
            # donate_argnums=0,
            # out_shardings=jtu.tree_map(lambda a: a.sharding, tstate),
        )
        tstate = p_train_step(tstate, inputs)
        loss = float(jax.device_get(tstate.loss))
        progress.start_task(task)
        progress.update(task, description="training...", loss=loss, advance=1)
        for inputs in it.islice(data, train_steps):
            tstate = p_train_step(tstate, inputs)
            loss = float(jax.device_get(tstate.loss))
            progress.update(task, description="training...", loss=loss, advance=1)

    pass


if __name__ == "__main__":
    fire.Fire(main)
