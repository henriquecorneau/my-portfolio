---
title: Generative models — diffusion
summary: From recognising images to creating them. Build a denoising diffusion model from scratch — forward noising, a network that predicts noise, and the reverse sampling loop — watch it turn Gaussian noise into a learnt distribution, then see how the same idea, moved into a latent space and conditioned on text, becomes Stable Diffusion.
topics: generative models · DDPM · noise schedule · denoising objective · ancestral sampling · latent diffusion · classifier-free guidance
script: 17_diffusion.py
sources: UFRN Deep Learning class 13 (generative AI, Stable Diffusion)
---

## Learning a distribution

Every model so far learnt $p(y \mid \mathbf{x})$: given an input, predict a label. A **generative model** learns $p(\mathbf{x})$ itself, well enough to **draw new samples** from it — new faces, new handwriting, new images that match a sentence. The difficulty is that $p(\mathbf{x})$ over images is an absurdly complicated distribution in a space with hundreds of thousands of dimensions.

Over the last decade the leading approaches were GANs (a generator against a discriminator), VAEs (an autoencoder with a probabilistic bottleneck) and **diffusion models**, which now power almost every image generator. Their idea is disarmingly simple: destroying data with noise is easy, so learn to *undo* it one small step at a time.

We work in 2-D, where the whole process fits in a scatter plot and trains on a CPU in half a minute: the target distribution is the two-moons shape.

```python
--8<-- "17_diffusion.py:data"
```

## The forward process: adding noise

Over $T$ steps, mix in a little Gaussian noise at each step: $\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}$, with a **noise schedule** $\beta_1 < \dots < \beta_T$. Because Gaussians compose, we can jump directly to any step:

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\epsilon}, \qquad \bar\alpha_t = \prod_{s \le t}(1 - \beta_s), \quad \boldsymbol\epsilon \sim \mathcal N(0, I)$$

```python
--8<-- "17_diffusion.py:config"
```

```python
--8<-- "17_diffusion.py:schedule"
```

```text
signal left: t=0 1.000, t=T/4 0.715, t=T/2 0.263, t=T 0.0046
```

After $T$ steps less than 0.5% of the signal is left: $\mathbf{x}_T$ is indistinguishable from pure noise. That is the point — we know how to sample pure noise.

!fig 17-forward | The forward process on the two moons: by t = 50 the shape is gone, at t = 99 it is a standard Gaussian.

## The model: predict the noise

To reverse a step we need to know which direction leads back to the data. The DDPM paper (Ho et al., 2020) showed that a good way to learn it is to train a network $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ to **predict the noise** that was added — a plain regression with the squared error of lesson 02:

$$L = \mathbb{E}_{\mathbf{x}_0, t, \boldsymbol\epsilon}\ \big\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ t)\big\|^2$$

The same network handles every noise level, so it receives $t$ as an input, encoded with sines and cosines of several frequencies (the trick transformers use for positions).

```python
--8<-- "17_diffusion.py:model"
```

```python
--8<-- "17_diffusion.py:train"
```

```text
training loss: first 100 steps 0.358, last 500 steps 0.235
```

The loss never goes to zero, and it shouldn't: at high noise levels the noise is genuinely unpredictable. What the network learns is the *expected* noise given $\mathbf{x}_t$, which is (up to a factor) the **score** $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ — the direction in which the noisy data density increases.

!fig 17-field | The negated noise prediction over a grid: at t = 40 it points roughly towards the centre of the data; at t = 5 it points precisely onto the two moons. Right: the training loss.

## Sampling: walk back from noise

Start from $\mathbf{x}_T \sim \mathcal N(0, I)$ and apply, for $t = T, \dots, 1$,

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)\Big) + \sqrt{\beta_t}\,\mathbf{z}, \qquad \mathbf{z} \sim \mathcal N(0, I)$$

remove a bit of the predicted noise, then add back a smaller amount of fresh noise (except at the last step).

```python
--8<-- "17_diffusion.py:sample"
```

```text
MMD generated vs real: 0.0074 | pure noise vs real: 0.0476 | two real halves: 0.0006
```

!fig 17-reverse | 2,000 samples during the reverse process. Pure noise at t = 99; the moons emerge in the last 25 steps.

How good are the samples? Evaluating generative models is hard — there is no label to compare with. Here we measure the **maximum mean discrepancy** between generated and real points: zero when the two sets come from the same distribution. Our samples are far closer to the data than noise is (0.007 vs 0.048), but not as close as two halves of the real data (0.0006): the moons come out slightly too thick. More training steps, a bigger network or more noise levels close the gap. For images the standard metric is the FID, the same idea computed on the features of a pretrained CNN.

## From two moons to Stable Diffusion

Three additions take this exact algorithm to text-to-image generation:

1. **A U-Net instead of an MLP.** For images, $\boldsymbol\epsilon_\theta$ is a convolutional encoder–decoder with skip connections (and attention layers), which predicts a noise *image* of the same size as its input.
2. **Diffusion in a latent space.** Running 50 network evaluations on a 512×512×3 image is expensive. *Latent diffusion* (Rombach et al., 2022) first trains an autoencoder that compresses an image 8× in each direction into a 64×64×4 latent, runs the diffusion there, and decodes the result once at the end.
3. **Conditioning on text.** A text encoder (CLIP) turns the prompt into a sequence of vectors that the U-Net reads through cross-attention, so it learns $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \text{text})$. At sampling time **classifier-free guidance** extrapolates from the unconditional prediction towards the conditional one, $\hat{\boldsymbol\epsilon} = \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \varnothing) + w\,(\boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \text{text}) - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \varnothing))$; the `guidance_scale` $w$ trades diversity for prompt adherence.

The UFRN class used the pretrained model through Hugging Face `diffusers`. The script below needs a GPU (a free Colab T4 is enough), so unlike the rest of the course it is **not** run when these pages are built:

```python
--8<-- "17b_stable_diffusion.py:pipeline"
```

!fig 17-stable-diffusion | What Stable Diffusion 2 produced in class for the prompt "dog in a human body with suit" (768×768, guidance scale 10, 50 steps). A convincing dog — no human body, no suit.

The picture is a good summary of what these models do well and badly. The image is clean and coherent, because the model learnt the distribution of images extremely well; the *composition* the prompt asked for is ignored, because a frozen CLIP text encoder captures "dog" and "suit" as concepts much better than how they should be combined. Newer models (larger text encoders, transformers instead of U-Nets) follow prompts much better — the diffusion part is still the same loop as above.

## Exercises

1. Reduce $T$ to 20 with the same $\beta_{\text{end}}$. What happens to the samples? Then try **DDIM** sampling: a deterministic update that skips steps (`x0_pred` from $\boldsymbol\epsilon_\theta$, then jump to $t - k$).
2. Make the model **conditional**: train on both moons with the moon index as an extra input (an embedding added to the time features), then generate only the upper moon. Add classifier-free guidance by dropping the label 10% of the time during training.
3. Train the same DDPM on 28×28 MNIST digits with a small U-Net (two downsampling and two upsampling blocks). On a CPU, keep it to a few epochs on one digit class.
