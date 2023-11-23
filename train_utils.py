from typing import NamedTuple
import jax.numpy as jnp
import jax
import diffusers
from einops import rearrange
import jax.tree_util as jtu
import optax
import flax.traverse_util as ftu
import scipy.stats as sst
from transformers import CLIPTokenizer
from istrm import RandomCrop, load_chunks
import polars as pl
import numpy as np
from diffusers import (
    FlaxDDPMScheduler,
    FlaxAutoencoderKL,
    FlaxUNet2DConditionModel,
)
from transformers import FlaxCLIPTextModel
from diffusers.models.modeling_flax_pytorch_utils import (
    convert_pytorch_state_dict_to_flax,
)


class Inputs(NamedTuple):
    images: jax.Array
    tokens: dict


class Pipeline(NamedTuple):
    clip: FlaxCLIPTextModel
    vae: FlaxAutoencoderKL
    unet: FlaxUNet2DConditionModel
    scheduler: FlaxDDPMScheduler
    skip_steps: int

    def init(model_slug, scheduler, skip_steps, dtype=jnp.bfloat16):
        def _load_pretrained_diffusers_ckpt(subfolder, flax_cls):
            pt_cls = getattr(diffusers, flax_cls.__name__.removeprefix("Flax"))
            pt_model = pt_cls.from_pretrained(model_slug, subfolder=subfolder)
            flax_model = flax_cls.from_config(pt_model.config, dtype=dtype)
            state = convert_pytorch_state_dict_to_flax(
                pt_model.state_dict(), flax_model
            )
            return flax_model, state

        clip = FlaxCLIPTextModel.from_pretrained(
            model_slug, subfolder="text_encoder", dtype=dtype
        )
        vae, vae_params = _load_pretrained_diffusers_ckpt("vae", FlaxAutoencoderKL)
        unet, unet_params = _load_pretrained_diffusers_ckpt(
            "unet", FlaxUNet2DConditionModel
        )
        params = {"clip": clip.params, "vae": vae_params, "unet": unet_params}
        return Pipeline(clip, vae, unet, scheduler, skip_steps), params


def step_weights(scheduler, skip_steps, timesteps):
    "generate the inverse-sigma weighting for the given timesteps"
    # wts = jax.scipy.stats.norm.sf(scale=sigma, x=jnp.log(t0s))
    noise_state = scheduler.create_state()
    wts = noise_state.common.betas
    wts = wts[skip_steps:] - wts[:-skip_steps]
    wts = jnp.cumsum(wts / wts.sum())[::-1]
    wts = wts[timesteps]
    return wts


def bound_interp(scheduler, skip_steps, x, y, t):
    assert x.shape == y.shape, "x and y must have the same shape"
    c_skip = step_weights(scheduler, skip_steps, t).reshape([-1] + [1] * (x.ndim - 1))
    c_out = 1 - c_skip
    return c_skip * x + c_out * y


def perturb(rng: jax.random.PRNGKey, latents: jax.Array, gamma: float):
    "https://arxiv.org/abs/2301.11706"
    alpha = np.sqrt(gamma)
    noise = jax.random.normal(rng, shape=latents.shape)
    return alpha * noise + (1 - alpha) * latents


def noisy_latents(params, vae, rng, images, gamma):
    sample_rng, perturb_rng = jax.random.split(rng)
    latents = vae.apply(
        {"params": params},
        images,
        method=vae.encode,
        deterministic=False,
    ).latent_dist.sample(sample_rng)
    latents = rearrange(latents, "... h w c -> ... c h w")
    latents *= vae.config.scaling_factor
    latents = perturb(perturb_rng, latents, gamma=gamma)
    return latents


def noise_and_denoise(params, pipe: Pipeline, rng, inputs, timesteps, gamma):
    "noise and denoise the given image"
    vae = pipe.vae
    images, tokens = inputs.images, inputs.tokens
    noise_key, sample_key = jax.random.split(rng)
    images = rearrange(images / 255.0, "... h w c -> ... c h w")
    latents = noisy_latents(params["vae"], pipe.vae, sample_key, images, gamma)
    noise_state = pipe.scheduler.create_state()
    eps = jax.random.normal(noise_key, latents.shape)
    # Get the text embedding for conditioning
    classifier_guidance = pipe.clip.module.apply(
        {"params": params["clip"]},
        **{**tokens, "position_ids": jnp.zeros_like(tokens["input_ids"])},
        deterministic=False,
        return_dict=True,
    ).last_hidden_state
    output_latents = pipe.unet.apply(
        {"params": params["unet"]},
        pipe.scheduler.add_noise(noise_state, latents, eps, timesteps),
        timesteps,
        classifier_guidance,
        train=True,
    ).sample
    output_latents = bound_interp(
        pipe.scheduler, pipe.skip_steps, latents, output_latents, timesteps
    )
    output_images = vae.apply(
        {"params": params["vae"]},
        output_latents,
        method=vae.decode,
    ).sample
    return output_images


def random_steps(scheduler, skip_steps, rng, shape):
    "generate log-normally distributed timesteps"
    sched_length = scheduler.config.num_train_timesteps - skip_steps
    sigma = np.log(np.sqrt(sched_length))
    times = jax.random.lognormal(rng, sigma=sigma, shape=shape) + skip_steps
    times = jnp.rint(times).astype(int)
    times = jnp.minimum(times, sched_length)
    return times


def input_stream(
    posts: pl.DataFrame,
    tokenizer: CLIPTokenizer,
    batch_size: int,
    cg_dropout: float,
    image_size=[512, 512],
    seed=0,
):
    shards = jax.sharding.PositionalSharding(jax.local_devices())

    def tokenize(captions):
        tokens = tokenizer(captions, padding="max_length", return_tensors="np")
        return dict(tokens)

    dropout_rng = np.random.default_rng(seed)

    views = RandomCrop(image_size, seed=dropout_rng.bit_generator.random_raw())
    for images, meta in load_chunks(posts, batch_size=batch_size, views=views):
        rating_names = [
            f"{rating}-rated"
            for rating in ("badly", "poorly", "unremarkably", "greatly", "excellently")
        ]
        scores = meta["score"].to_numpy().astype(np.float32)
        rating_bins = np.minimum(
            len(rating_names) - 1,
            np.rint(sst.norm.cdf(scores) * len(rating_names)).astype(int),
        )
        aesthetic_prefixes = [rating_names[i] for i in rating_bins]
        exploded = meta.explode("caption")
        exploded = exploded.sample(fraction=0.9, shuffle=True)
        captions = (
            exploded.group_by("id")
            .agg(pl.exclude("caption").first(), "caption")
            .with_columns(pl.col("caption").list.join(", "))
            .to_series()
        )
        prefixed_captions = []
        for aesthetic_prefix, caption in zip(aesthetic_prefixes, captions):
            prefix = "" if dropout_rng.uniform() < cg_dropout else aesthetic_prefix
            if dropout_rng.uniform() > cg_dropout:
                prefixed_captions.append(f"{prefix}, {caption}")
            else:
                prefixed_captions.append(prefix)
        inputs = Inputs(
            images=images,
            tokens=tokenize(prefixed_captions),
        )
        in_sharding = jtu.tree_map(
            lambda a: shards.reshape([-1] + [1] * (a.ndim - 1)), inputs
        )
        inputs = jax.device_put(inputs, in_sharding)
        yield inputs


def drop_posemb(params):
    path = ("clip", "text_model", "embeddings", "position_embedding", "embedding")
    params = ftu.flatten_dict(params)
    params[path] = jnp.zeros_like(params[path])
    params = ftu.unflatten_dict(params)
    return params


def merge_params(*trees):
    output = dict()
    for tree in trees:
        output |= ftu.flatten_dict(tree)
    return ftu.unflatten_dict(output)


def objective(
    online: optax.Params,
    target: optax.Params,
    frozen: optax.Params,
    pipe: Pipeline,
    rng: jax.random.PRNGKey,
    inputs: Inputs,
    gamma: float = 0.0,  # TODO: test with gamma=0.0
):
    denoising_key, timestep_key = jax.random.split(rng)
    online = drop_posemb(merge_params(online, frozen))
    target = drop_posemb(merge_params(target, frozen))
    t0s = random_steps(
        pipe.scheduler,
        pipe.skip_steps,
        timestep_key,
        shape=inputs.images.shape[:-3],
    )
    t1s = t0s + pipe.skip_steps
    ft0 = noise_and_denoise(target, pipe, denoising_key, inputs, t0s, gamma)
    ft1 = noise_and_denoise(online, pipe, denoising_key, inputs, t1s, gamma)
    yiq = [[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]]
    yiq = jnp.array(yiq)
    ft0, ft1 = map(lambda x: rearrange(x, "... c h w -> ... h w c") @ yiq, (ft0, ft1))
    # delta taken from Improved Techniques
    err = optax.huber_loss(ft0, ft1, delta=0.03)
    # luminance (texture) is more perceptually important than chrominance (color)
    err = err.at[..., 1:].mul(0.5)
    wts = step_weights(pipe.scheduler, pipe.skip_steps, t0s)
    wts = wts.reshape([-1] + [1] * (ft0.ndim - 1))
    return jnp.mean(wts * err)
