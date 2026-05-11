# ising-sampling

Note: I used LLMs to hash out unfamiliar physics concepts and polish the markdown prose/code implementation.

I built a small autoregressive transformer that learns to approximately sample a 16x16 Ising lattice (for quicker training) at the critical temperature (2.269). The target distribution is the standard 2D nearest-neighbor Ising model with periodic boundary conditions:

$$
E(x) = -\sum_{\langle i,j\rangle} x_i x_j
$$

with spins $x_i \in \{-1, +1\}$ and coupling strength/Boltzmann constant = 1.

I used a transformer for language-model motivation of sparsity and mean-field structure. A CNN may be more nautral given the Ising model is local and spatial, but I was curious to see whether a sequence model could learn the flattened physical system's distribution well enough to reproduce its characteristics. Each lattice is flattened into 256 binary spin tokens, where -1 = 0 and +1 = 1. The model learns the usual autoregressive factorization:

$$
p(x) = \prod_t p(x_t \mid x_1,\ldots,x_{t-1})
$$

To avoid the flattening being completely geometry-blind, I add row/col embeddings alongside the spin-token embedding. Training is standard shifted next-token prediction. I use a Wolff cluster sampler at the critical temperature to generate training data (as opposed to single-spin Metropolis), assuming that critical temp gives rise to clusters in the lattice.

For evaluation, I score the sampler using three statistics: 1) energy, 2) absolute magnetization, and 3) the spin-spin correlation function.

Energy per site evaluates feasibility of local nearest-neighbor statistics (generated samples with incorrect energy distribution = model hasn't learned local physics). Absolute magnetization evaluates finite-size distribution between order/disorder. Then, importantly is the two-point correlation function:

$$
C(r) = \mathbb{E}[x_i x_{i+r}]
$$

The correlation curve evaluates whether the model captures the fact that critical samples contain structure across many distances. For each statistic, I compare generated samples to held-out Wolff samples, then normalize that error by the error from a random-spin baseline. The final score is a normalized value between 0–100 that measures performance against a trivial sampler (rather than reporting an uncalibrated distance).

I heuristically combine the normalized errors as:

$$
\text{error} = 0.25q_E + 0.25q_M + 0.50q_C
$$

and report score = 100exp(−error).

If the model matches energy but not correlations, it learned something like local Ising but not the critical temp characteristics. If it matches magnetization but not energy, it did not find the correct local interactions. I do not expect this small transformer to be the best possible Ising sampler (a convolutional or denoising architecture may be more sample-efficient). However, if it can effectively approximate this distribution then we can readily ask questions about sparse feature activations in language models. What are the effective variables/couplings, where might mean field be relevant, and where are higher-order/frustrated interactions?

Repo: 

`src/data.py` = generate critical Ising samples

`src/model.py` = transformer

`src/train.py` = train

`src/metrics.py` = evaluation stats

`src/evaluate.py` = sample from trained model

To run:

```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

A quick run (20,000 training samples, 1 epoch, 256 generated evaluation samples) gives:

```
Sampler score: 92.74
energy_error_normalized:        0.078
magnetization_error_normalized: 0.057
correlation_error_normalized:   0.083
```

The model is much closer to held-out Wolff samples than to random-spin samples on all three metrics.
