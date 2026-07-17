# Auto-Encoding Variational Bayes — Compact Study Notes
**Kingma & Welling, 2013 (arXiv:1312.6114)**

> Goal of this document: every equation, definition, and result from the paper, compressed into a fast-to-read reference. Nothing dropped — just stripped of redundant prose.

---

## 1. The Problem

We have a directed generative model:

$$z^{(i)} \sim p_{\theta^*}(z), \qquad x^{(i)} \sim p_{\theta^*}(x \mid z)$$

i.e. a latent variable $z$ generates an observation $x$. We see $X = \{x^{(i)}\}_{i=1}^N$ but **not** $z^{(i)}$ or the true parameters $\theta^*$.

**Graphical model:**

```
        θ
        │
   ┌────▼────┐         φ
   │ z ──────┼───►x◄───┘
   └─────────┘
   (solid: generative pθ(z)pθ(x|z))
   (dashed: recognition qφ(z|x))
```

We want to handle the *general, intractable* case:

1. **Intractability** — the marginal $p_\theta(x) = \int p_\theta(z)p_\theta(x\mid z)\,dz$ is intractable, so is the true posterior $p_\theta(z\mid x) = p_\theta(x\mid z)p_\theta(z)/p_\theta(x)$ → EM is out, mean-field VB integrals are also intractable. Common whenever $p_\theta(x|z)$ involves a nonlinear neural net.
2. **Large datasets** — batch optimization too expensive; need minibatch-based, online updates (rules out expensive per-datapoint sampling like MCMC-EM).

**Three goals:**
| Goal | Use case |
|---|---|
| Efficient approximate ML/MAP estimation of $\theta$ | Model analysis, data generation |
| Efficient approximate posterior inference of $z$ given $x,\theta$ | Coding / representation |
| Efficient approximate marginal inference of $x$ | Denoising, inpainting, super-resolution |

**Key idea — the recognition model.** Introduce $q_\phi(z\mid x)$: an approximate posterior, *not* required to factorize, with $\phi$ learned **jointly** with $\theta$ (no closed-form coordinate updates as in mean-field VB).

- $q_\phi(z\mid x)$ = **probabilistic encoder** (datapoint → distribution over codes)
- $p_\theta(x\mid z)$ = **probabilistic decoder** (code → distribution over datapoints)

---

## 2. The Variational Bound

Per-datapoint marginal log-likelihood decomposes exactly:

$$\log p_\theta(x^{(i)}) = D_{KL}\big(q_\phi(z|x^{(i)}) \,\|\, p_\theta(z|x^{(i)})\big) + \mathcal{L}(\theta,\phi;x^{(i)}) \tag{1}$$

Since KL $\ge 0$, $\mathcal{L}$ is a **lower bound** on the log marginal likelihood ("ELBO"):

$$\log p_\theta(x^{(i)}) \ge \mathcal{L}(\theta,\phi;x^{(i)}) = \mathbb{E}_{q_\phi(z|x)}\big[-\log q_\phi(z|x) + \log p_\theta(x,z)\big] \tag{2}$$

Equivalently, splitting into a regularizer + reconstruction term:

$$\mathcal{L}(\theta,\phi;x^{(i)}) = -D_{KL}\big(q_\phi(z|x^{(i)})\,\|\,p_\theta(z)\big) + \mathbb{E}_{q_\phi(z|x^{(i)})}\big[\log p_\theta(x^{(i)}|z)\big] \tag{3}$$

**Why not just do plain Monte Carlo on $\nabla_\phi$?** The naive REINFORCE-style estimator

$$\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}\Big[f(z)\nabla_{q_\phi(z)}\log q_\phi(z)\Big] \approx \frac{1}{L}\sum_{l=1}^L f(z)\nabla_{q_\phi(z^{(l)})}\log q_\phi(z^{(l)})$$

has **very high variance** → impractical. This motivates the reparameterization trick.

---

## 3. The Reparameterization Trick (Section 2.4)

**Core idea.** Instead of sampling $z \sim q_\phi(z|x)$ directly, express $z$ as a *deterministic, differentiable* function of $x$ and an independent noise variable $\varepsilon$:

$$\tilde z = g_\phi(\varepsilon, x), \qquad \varepsilon \sim p(\varepsilon) \tag{4}$$

**Why this works (proof sketch).** Given the deterministic map $z = g_\phi(\varepsilon,x)$:
$$q_\phi(z|x)\prod_i dz_i = p(\varepsilon)\prod_i d\varepsilon_i$$
so
$$\int q_\phi(z|x) f(z)\,dz = \int p(\varepsilon) f(g_\phi(\varepsilon,x))\,d\varepsilon \approx \frac{1}{L}\sum_{l=1}^L f(g_\phi(x,\varepsilon^{(l)})), \quad \varepsilon^{(l)}\sim p(\varepsilon) \tag{5}$$

Now the sampling randomness ($\varepsilon$) is **independent of $\phi$**, so the Monte Carlo estimate is differentiable w.r.t. $\phi$ via standard backprop — this is the whole trick.

**Canonical example — univariate/multivariate Gaussian:**
$$z \sim \mathcal{N}(\mu,\sigma^2) \quad\Longleftrightarrow\quad z = \mu + \sigma\varepsilon,\ \ \varepsilon\sim\mathcal{N}(0,1)$$
$$\mathbb{E}_{\mathcal{N}(z;\mu,\sigma^2)}[f(z)] = \mathbb{E}_{\mathcal{N}(\varepsilon;0,1)}[f(\mu+\sigma\varepsilon)] \approx \frac{1}{L}\sum_{l=1}^L f(\mu+\sigma\varepsilon^{(l)})$$

**Three general strategies to find $(p(\varepsilon), g_\phi)$:**

1. **Tractable inverse CDF**: $\varepsilon\sim U(0,I)$, $g_\phi$ = inverse CDF of $q_\phi(z|x)$.
   *Examples:* Exponential, Cauchy, Logistic, Rayleigh, Pareto, Weibull, Reciprocal, Gompertz, Gumbel, Erlang.
2. **Location–scale family**: take standard form ($\text{loc}=0,\text{scale}=1$) as $\varepsilon$, then $g(\cdot)=\text{location}+\text{scale}\cdot\varepsilon$.
   *Examples:* Laplace, Elliptical, Student's-t, Logistic, Uniform, Triangular, **Gaussian**.
3. **Composition**: express $z$ as a transformation of other, simpler-to-reparameterize variables.
   *Examples:* Log-Normal (exp of Normal), Gamma (sum of Exponentials), Dirichlet (weighted sum of Gammas), Beta, Chi-Squared, F.

If all fail, approximate the inverse CDF (cost comparable to evaluating the PDF).

---

## 4. The SGVB Estimator and AEVB Algorithm

Applying the trick to eq. (2) gives the generic **SGVB estimator** $\tilde{\mathcal{L}}^A \approx \mathcal{L}$:

$$\tilde{\mathcal{L}}^A(\theta,\phi;x^{(i)}) = \frac{1}{L}\sum_{l=1}^L \log p_\theta(x^{(i)}, z^{(i,l)}) - \log q_\phi(z^{(i,l)}|x^{(i)}), \quad z^{(i,l)}=g_\phi(\varepsilon^{(i,l)},x^{(i)}) \tag{6}$$

**Lower-variance version.** When $D_{KL}(q_\phi(z|x^{(i)})\|p_\theta(z))$ is analytically integrable (true for Gaussian prior + Gaussian posterior — see §6), only the reconstruction term needs MC sampling. This gives estimator $\tilde{\mathcal{L}}^B$, corresponding to eq. (3):

$$\tilde{\mathcal{L}}^B(\theta,\phi;x^{(i)}) = -D_{KL}\big(q_\phi(z|x^{(i)})\,\|\,p_\theta(z)\big) + \frac{1}{L}\sum_{l=1}^L \log p_\theta(x^{(i)}|z^{(i,l)}) \tag{7}$$

The KL term acts as a **regularizer** pulling $q_\phi$ toward the prior; the second term is **expected reconstruction error**. $\tilde{\mathcal{L}}^B$ typically has lower variance than $\tilde{\mathcal{L}}^A$.

**Minibatch estimator for the full dataset** (size $N$, minibatch size $M$):

$$\mathcal{L}(\theta,\phi;X) \approx \tilde{\mathcal{L}}^M(\theta,\phi;X^M) = \frac{N}{M}\sum_{i=1}^M \tilde{\mathcal{L}}(\theta,\phi;x^{(i)}) \tag{8}$$

In practice: **$L=1$ sample per datapoint suffices** as long as minibatch size $M$ is large enough (paper uses $M=100$).

### Algorithm 1 — Minibatch AEVB
```
θ, φ ← initialize parameters
repeat
    X_M  ← random minibatch of M datapoints
    ε    ← random samples from noise prior p(ε)
    g    ← ∇_{θ,φ} L̃^M(θ, φ; X_M, ε)      # gradient of eq. (8)
    θ, φ ← update via SGD / Adagrad using g
until convergence
return θ, φ
```

**The autoencoder connection.** In eq. (7): term 1 (KL from prior) = regularizer; term 2 = negative reconstruction error. $g_\phi$ maps $(x^{(i)}, \varepsilon^{(l)}) \to z^{(i,l)} \sim q_\phi(z|x^{(i)})$ ("encode"), then $\log p_\theta(x^{(i)}|z^{(i,l)})$ scores how well $z$ reconstructs $x$ ("decode"). When the encoder/decoder are neural nets → **Variational Auto-Encoder (VAE)**.

---

## 5. Worked Example: The Variational Auto-Encoder (Section 3)

**Setup:**
- Prior: centered isotropic Gaussian, **no parameters**: $p_\theta(z) = \mathcal{N}(z; 0, I)$
- Decoder $p_\theta(x|z)$: Gaussian (real-valued data) or Bernoulli (binary data), parameters computed from $z$ via an MLP
- Encoder (approximate posterior), assumed Gaussian with diagonal covariance:

$$\log q_\phi(z|x^{(i)}) = \log \mathcal{N}\big(z;\, \mu^{(i)}, \sigma^{2(i)}I\big) \tag{9}$$

where $\mu^{(i)}, \sigma^{(i)}$ are **outputs of the encoder MLP** — nonlinear functions of $x^{(i)}$ and $\phi$.

**Reparameterized sampling:**
$$z^{(i,l)} = g_\phi(x^{(i)}, \varepsilon^{(l)}) = \mu^{(i)} + \sigma^{(i)} \odot \varepsilon^{(l)}, \qquad \varepsilon^{(l)}\sim\mathcal{N}(0,I)$$
($\odot$ = elementwise product.)

**Since both prior and posterior are Gaussian, the KL term is solved analytically (derivation in §6 below). Final per-datapoint estimator:**

$$\mathcal{L}(\theta,\phi;x^{(i)}) \approx \frac{1}{2}\sum_{j=1}^J\Big(1+\log\big((\sigma_j^{(i)})^2\big) - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2\Big) \;+\; \frac{1}{L}\sum_{l=1}^L \log p_\theta(x^{(i)}|z^{(i,l)}) \tag{10}$$

where $z^{(i,l)} = \mu^{(i)} + \sigma^{(i)}\odot\varepsilon^{(l)}$, $J$ = latent dimensionality.

This is **exactly the standard VAE loss**: *KL-to-prior regularizer* + *reconstruction log-likelihood* (Bernoulli cross-entropy or Gaussian negative-MSE-like term, depending on data type).

---

## 6. Analytic KL Divergence (Appendix B) — derivation

For $J$-dimensional $z$, prior $p_\theta(z)=\mathcal{N}(0,I)$, posterior $q_\phi(z|x^{(i)}) = \mathcal{N}(\mu,\sigma^2 I)$:

$$\int q_\theta(z)\log p(z)\,dz = -\frac{J}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^J(\mu_j^2+\sigma_j^2)$$

$$\int q_\theta(z)\log q_\theta(z)\,dz = -\frac{J}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^J(1+\log\sigma_j^2)$$

Subtracting (these are $-\text{cross-entropy}$ and $-\text{entropy}$ respectively):

$$\boxed{-D_{KL}\big(q_\phi(z)\,\|\,p_\theta(z)\big) = \frac{1}{2}\sum_{j=1}^J\Big(1+\log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\Big)} \tag{B}$$

This is the closed form plugged directly into eq. (10).

---

## 7. Neural Network Architectures (Appendix C)

### Bernoulli MLP decoder (binary data)
$$\log p(x|z) = \sum_{i=1}^D x_i\log y_i + (1-x_i)\log(1-y_i), \qquad y = f_\sigma\big(W_2\tanh(W_1 z+b_1)+b_2\big) \tag{11}$$
($f_\sigma$ = elementwise sigmoid; $\theta=\{W_1,W_2,b_1,b_2\}$.)

### Gaussian MLP encoder/decoder (real-valued data, or the encoder in general)
$$\log p(x|z) = \log\mathcal{N}(x;\mu,\sigma^2 I)$$
$$\mu = W_4 h + b_4,\qquad \log\sigma^2 = W_5h+b_5,\qquad h=\tanh(W_3z+b_3) \tag{12}$$

When used as **encoder** $q_\phi(z|x)$: swap roles of $x,z$; weights/biases become $\phi$ instead of $\theta$.

Both encoder and decoder are simple single-hidden-layer MLPs with matching hidden-unit counts in the experiments.

---

## 8. Experiments

**Datasets:** MNIST, Frey Face. **Setup:** $M=100$, $L=1$; params init $\sim\mathcal{N}(0,0.01)$; Adagrad stepsize ∈ {0.01, 0.02, 0.1}; weight decay ≈ MAP prior $p(\theta)=\mathcal{N}(0,I)$.

**Baseline compared against:** the **wake-sleep algorithm** (only other applicable online method in the literature at the time).

**Result 1 — Lower bound convergence (Fig. 2):** AEVB converges **considerably faster** and reaches a **better solution** than wake-sleep across all latent dimensionalities tested ($N_z \in \{3,5,10,20,200\}$ for MNIST; $\{2,5,10,20\}$ for Frey Face). Notably: **more latent variables did not cause more overfitting** — explained by the regularizing effect of the KL term in the bound.

**Result 2 — Marginal likelihood (Fig. 3):** For low-dim latent space ($N_z=3$, 100 hidden units), comparing AEVB vs. wake-sleep vs. **Monte Carlo EM (MCEM) with Hybrid Monte Carlo**: AEVB matches or beats both, and MCEM **isn't a practical online algorithm** for the full MNIST set (too slow at scale).

**Result 3 — Visualization (Appendix A, Figs 4–5):** With a 2-D latent space, the learned encoder gives a smooth 2-D manifold of MNIST digits / Frey faces (latent coordinates mapped through inverse-Gaussian-CDF on a unit-square grid, each point decoded back to image space). Higher-dim latent spaces (5/10/20-D) sampled randomly also produce coherent digit images.

---

## 9. Marginal Likelihood Estimator (Appendix D)

For low-dimensional latent space ($<5$D), to actually *estimate* $p_\theta(x^{(i)})$ (not just the bound):

1. Sample $L$ values $\{z^{(l)}\}$ from the posterior via gradient-based MCMC (HMC), using
 $$\nabla_z\log p_\theta(z|x) = \nabla_z\log p_\theta(z) + \nabla_z\log p_\theta(x|z)$$
2. Fit a density estimator $q(z)$ to these samples.
3. Draw fresh posterior samples and plug into:
$$p_\theta(x^{(i)}) \approx \left(\frac{1}{L}\sum_{l=1}^L \frac{q(z^{(l)})}{p_\theta(z)p_\theta(x^{(i)}|z^{(l)})}\right)^{-1}, \qquad z^{(l)}\sim p_\theta(z|x^{(i)}) \tag{D}$$

**Derivation:**
$$\frac{1}{p_\theta(x^{(i)})} = \int \frac{q(z)}{p_\theta(x^{(i)})}\,dz = \int p_\theta(z|x^{(i)})\frac{q(z)}{p_\theta(x^{(i)},z)}\,dz \approx \frac{1}{L}\sum_l \frac{q(z^{(l)})}{p_\theta(z)p_\theta(x^{(i)}|z^{(l)})}$$

**Monte Carlo EM (Appendix E):** no encoder at all — samples posterior directly via HMC (10 leapfrog steps, auto-tuned step size, 90% acceptance), then does 5 weight-update steps per sampling round. Not online → can't scale to full datasets efficiently (confirmed in Fig. 3).

---

## 10. Full VB Extension — Variational Inference over $\theta$ Too (Appendix F)

Everything above does **point estimation** (ML/MAP) for $\theta$ and variational inference only for $z$. The appendix extends SGVB to put a variational posterior on the **parameters $\theta$ as well**.

**Setup.** Hyperprior $p_\alpha(\theta)$. Decompose:
$$\log p_\alpha(X) = D_{KL}\big(q_\phi(\theta)\|p_\alpha(\theta|X)\big) + \mathcal{L}(\phi;X) \tag{13}$$
$$\mathcal{L}(\phi;X) = \int q_\phi(\theta)\big(\log p_\theta(X) + \log p_\alpha(\theta) - \log q_\phi(\theta)\big)\,d\theta \tag{14}$$
and as before, per-datapoint:
$$\log p_\theta(x^{(i)}) = D_{KL}(q_\phi(z|x^{(i)})\|p_\theta(z|x^{(i)})) + \mathcal{L}(\theta,\phi;x^{(i)}) \tag{15}$$
$$\mathcal{L}(\theta,\phi;x^{(i)}) = \int q_\phi(z|x)\big(\log p_\theta(x^{(i)}|z)+\log p_\theta(z)-\log q_\phi(z|x)\big)\,dz \tag{16}$$

**Double reparameterization.** Reparameterize **both** $z$ and $\theta$:
$$\tilde z = g_\phi(\varepsilon,x),\ \varepsilon\sim p(\varepsilon) \quad\Rightarrow\quad \mathcal{L}(\theta,\phi;x^{(i)}) = \int p(\varepsilon)\Big[\log p_\theta(x^{(i)}|z)+\log p_\theta(z)-\log q_\phi(z|x)\Big]_{z=g_\phi(\varepsilon,x^{(i)})}d\varepsilon \tag{18}$$
$$\tilde\theta = h_\phi(\zeta),\ \zeta\sim p(\zeta) \quad\Rightarrow\quad \mathcal{L}(\phi;X) = \int p(\zeta)\Big[\log p_\theta(X)+\log p_\alpha(\theta)-\log q_\phi(\theta)\Big]_{\theta=h_\phi(\zeta)}d\zeta \tag{20}$$

Define shorthand:
$$f_\phi(x,z,\theta) = N\cdot\big(\log p_\theta(x|z)+\log p_\theta(z)-\log q_\phi(z|x)\big) + \log p_\alpha(\theta) - \log q_\phi(\theta) \tag{21}$$

**Combined Monte Carlo estimator:**
$$\mathcal{L}(\phi;X) \approx \frac{1}{L}\sum_{l=1}^L f_\phi\big(x^{(l)}, g_\phi(\varepsilon^{(l)},x^{(l)}), h_\phi(\zeta^{(l)})\big), \qquad \varepsilon^{(l)}\sim p(\varepsilon),\ \zeta^{(l)}\sim p(\zeta) \tag{22}$$

Differentiable w.r.t. $\phi$ since $\varepsilon,\zeta$ don't depend on $\phi$.

### Algorithm 2 — Full-VB stochastic gradient
```
Require: φ
g ← 0
for l = 1 to L:
    x ← random draw from X
    ε ← random draw from p(ε)
    ζ ← random draw from p(ζ)
    g ← g + (1/L)·∇_φ f_φ(x, g_φ(ε,x), h_φ(ζ))
return g
```

**Worked sub-example (all-Gaussian case, F.1):**
$$p_\alpha(\theta)=\mathcal{N}(0,I),\quad p_\theta(z)=\mathcal{N}(0,I),\quad q_\phi(\theta)=\mathcal{N}(\mu_\theta,\sigma_\theta^2I),\quad q_\phi(z|x)=\mathcal{N}(\mu_z,\sigma_z^2I)$$
Reparameterize:
$$\tilde\theta = \mu_\theta+\sigma_\theta\odot\zeta,\ \zeta\sim\mathcal{N}(0,I) \qquad \tilde z = \mu_z+\sigma_z\odot\varepsilon,\ \varepsilon\sim\mathcal{N}(0,I)$$
Four of the terms in $f_\phi$ are now analytic (both KLs solvable per §6), giving the lower-variance estimator:

$$\mathcal{L}(\phi;X) \approx \frac{1}{L}\sum_{l=1}^L N\cdot\left[\frac{1}{2}\sum_{j=1}^J\Big(1+\log(\sigma^{(l)2}_{z,j})-\mu^{(l)2}_{z,j}-\sigma^{(l)2}_{z,j}\Big)+\log p_\theta(x^{(i)}|z^{(i)})\right] + \frac{1}{2}\sum_{j=1}^J\Big(1+\log(\sigma^{(l)2}_{\theta,j})-\mu^{(l)2}_{\theta,j}-\sigma^{(l)2}_{\theta,j}\Big) \tag{24}$$

*(Left to future work experimentally — the paper notes the algorithm is given but not tested in this setting.)*

---

## 11. Related Work — quick map

| Method | Relation to AEVB |
|---|---|
| **Wake-Sleep** [HDFN95] | Only other online method for this model class at the time. Also uses a recognition model, but optimizes *two separate* objectives that don't jointly bound the marginal likelihood. Applies to discrete latents too (AEVB as presented doesn't). Same per-datapoint complexity as AEVB. |
| **Stochastic Variational Inference** [HBWP13], control-variate schemes [BJP12], [RGB13] | Address the same high-variance naive gradient problem, but via control variates rather than reparameterization. |
| [SK13] | Similar reparameterization idea, applied to exponential-family natural-parameter VI. |
| **PCA** [Row98] | Shown to be the ML solution of a linear-Gaussian special case ($p(z)=\mathcal{N}(0,I)$, $p(x|z)=\mathcal{N}(Wz,\epsilon I)$, $\epsilon\to 0$) — historical link between linear autoencoders and this class of generative model. |
| **Autoencoder theory** [VLL+10] | Unregularized AE training ≈ maximizing a mutual-information lower bound (infomax) — but reconstruction alone isn't sufficient for useful representations [BCV13]; needs denoising/contractive/sparse regularization. SGVB's KL term *is* a principled regularizer, replacing ad-hoc regularization hyperparameters. |
| **DARN** [GMW13] | Also auto-encoder-structured directed model, but for **binary** latents. |
| [RMW14] (Rezende, Mohamed, Wierstra) | **Independently developed** the same reparameterization-based connection (auto-encoders ↔ directed graphical models ↔ SVI) — concurrent, complementary work. |
| GSNs [BTL13], PSD [KRL08], DBM recognition models [SL10] | Related encoder/decoder-style architectures, but for undirected (Boltzmann-machine-like) models or sparse coding, not general directed graphical models. |

---

## 12. Conclusion & Future Work (verbatim structure, condensed)

**Contributions:**
1. **SGVB** — a new, simple, differentiable, unbiased lower-bound estimator via reparameterization, usable with any continuous latent-variable model satisfying mild differentiability conditions.
2. **AEVB** — for i.i.d. data with per-datapoint continuous latents, a learning algorithm that uses SGVB to fit a recognition model, giving fast ancestral-sampling-based inference (no per-datapoint iterative scheme like MCMC needed).

**Future directions named in the paper:**
- (i) Deep/hierarchical architectures (e.g. convolutional nets) for encoder/decoder
- (ii) Time-series / dynamic Bayesian network extensions
- (iii) Applying SGVB to global parameters (this is exactly Appendix F, "Full VB")
- (iv) Supervised models with latent variables (for learning complex noise distributions)

---

## 13. One-Page Cheat Sheet

```
Generative model:     z ~ pθ(z) = N(0, I)         (no learned prior params)
                       x ~ pθ(x|z)                 (decoder, MLP)
Recognition model:     z ~ qφ(z|x) = N(μ(x), σ²(x)I)   (encoder, MLP)

Bound:  log pθ(x) ≥ L(θ,φ;x) = -DKL(qφ(z|x) || pθ(z)) + E_qφ[log pθ(x|z)]
                                  \_____________________/   \______________/
                                       regularizer             reconstruction

Reparameterize:  z = μ + σ ⊙ ε,   ε ~ N(0,I)     ← makes ∇φ tractable via backprop

Analytic KL (Gaussian-to-standard-Gaussian), per dim j=1..J:
   -DKL = ½ Σⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)

Per-datapoint loss to MAXIMIZE:
   L(θ,φ;x⁽ⁱ⁾) ≈ ½ Σⱼ(1+log σⱼ² - μⱼ² - σⱼ²)  +  (1/L)Σₗ log pθ(x⁽ⁱ⁾|z⁽ⁱ,ˡ⁾)

Training: minibatch SGD/Adagrad, M=100, L=1 sample/datapoint is enough.
```

---

### Notation Reference
| Symbol | Meaning |
|---|---|
| $x$ | observed data |
| $z$ | continuous latent variable |
| $\theta$ | generative model parameters |
| $\phi$ | variational (recognition/encoder) parameters |
| $q_\phi(z\lvert x)$ | approximate posterior / encoder |
| $p_\theta(x\lvert z)$ | likelihood / decoder |
| $p_\theta(z)$ | prior over latents |
| $\mathcal{L}(\theta,\phi;x)$ | variational lower bound (ELBO) for datapoint $x$ |
| $D_{KL}$ | Kullback–Leibler divergence |
| $\varepsilon,\zeta$ | auxiliary noise variables, independent of $\phi$ |
| $g_\phi,\, h_\phi$ | reparameterization functions for $z$, $\theta$ resp. |
| $M$ | minibatch size |
| $L$ | number of MC samples per datapoint |
| $N$ | full dataset size |
| $J$ | latent space dimensionality |
