# morie.fairness -- native GAN module (rootcoder007/morie)
"""GANs for spatial synthesis and CTGAN-style debiasing, implemented
natively: pure-Python MLPs with hand-derived backpropagation, a
hand-written Adam, and morie's own RNG. No JAX, no torch, no numpy.

The training objective is the non-saturating GAN of Goodfellow et al.
(2014, "Generative Adversarial Nets", NeurIPS, sec. 3): the
discriminator minimises binary cross-entropy on real-vs-fake logits
and the generator maximises log D(G(z)). Adam follows Kingma & Ba
(2015, "Adam: A Method for Stochastic Optimization", ICLR, alg. 1).
Backprop through the leaky-ReLU MLP is the standard reverse-mode
chain rule (Rumelhart, Hinton & Williams 1986).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np

__all__ = ["SpatialGAN", "CTGANDebiaser"]

_LEAK = 0.2


# ── tiny MLP: forward + hand-derived backward ────────────────────────


def _init_mlp(rng, sizes):
    """He-initialised MLP parameters as a list of ``(W, b)`` pairs."""
    params = []
    for i in range(len(sizes) - 1):
        scale = math.sqrt(2.0 / sizes[i])
        W = [[scale * float(v) for v in
              rng.normal(0, 1, sizes[i + 1])._flat()]
             for _ in range(sizes[i])]
        b = [0.0] * sizes[i + 1]
        params.append((W, b))
    return params


def _forward(params, X):
    """Forward pass; caches pre-activations for backprop.

    Returns (output, cache) where cache[l] = (input_l, pre_act_l).
    Hidden layers use leaky ReLU; the output layer is linear.
    """
    cache = []
    h = X
    L = len(params)
    for li, (W, b) in enumerate(params):
        n_in = len(W)
        n_out = len(b)
        pre = [[b[j] + sum(h[r][i] * W[i][j] for i in range(n_in))
                for j in range(n_out)] for r in range(len(h))]
        cache.append((h, pre))
        if li < L - 1:
            h = [[v if v > 0 else _LEAK * v for v in row]
                 for row in pre]
        else:
            h = pre
    return h, cache


def _backward(params, cache, dout):
    """Gradients of the loss w.r.t. every (W, b), given dL/d(output).

    Also returns dL/d(input) so a generator can chain through a frozen
    discriminator.
    """
    grads = [None] * len(params)
    delta = dout
    for li in range(len(params) - 1, -1, -1):
        W, b = params[li]
        h_in, pre = cache[li]
        n_in, n_out, B = len(W), len(b), len(h_in)
        if li < len(params) - 1:
            # activation derivative of THIS layer's output was applied
            # by the layer above; here delta is already d/d(post-act),
            # convert to d/d(pre-act) via leaky slope
            delta = [[d * (1.0 if p > 0 else _LEAK)
                      for d, p in zip(drow, prow)]
                     for drow, prow in zip(delta, pre)]
        gW = [[sum(h_in[r][i] * delta[r][j] for r in range(B))
               for j in range(n_out)] for i in range(n_in)]
        gb = [sum(delta[r][j] for r in range(B))
              for j in range(n_out)]
        grads[li] = (gW, gb)
        if li > 0:
            delta = [[sum(delta[r][j] * W[i][j]
                          for j in range(n_out))
                      for i in range(n_in)] for r in range(B)]
    din = [[sum(delta[r][j] * params[0][0][i][j]
                for j in range(len(params[0][1])))
            for i in range(len(params[0][0]))]
           for r in range(len(delta))] if False else None
    return grads, delta


def _input_grad(params, cache, dout):
    """dL/d(input of the network) — chain rule all the way down."""
    delta = dout
    for li in range(len(params) - 1, -1, -1):
        W, b = params[li]
        h_in, pre = cache[li]
        n_in, n_out, B = len(W), len(b), len(h_in)
        if li < len(params) - 1:
            delta = [[d * (1.0 if p > 0 else _LEAK)
                      for d, p in zip(drow, prow)]
                     for drow, prow in zip(delta, pre)]
        delta = [[sum(delta[r][j] * W[i][j] for j in range(n_out))
                  for i in range(n_in)] for r in range(B)]
    return delta


def _sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


# ── hand-written Adam (Kingma & Ba 2015, alg. 1) ─────────────────────


def _adam_init(params):
    st = []
    for W, b in params:
        st.append(([[0.0] * len(W[0]) for _ in W], [0.0] * len(b),
                   [[0.0] * len(W[0]) for _ in W], [0.0] * len(b)))
    return st


def _adam_step(params, grads, state, step, lr, b1=0.9, b2=0.999,
               eps=1e-8):
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    for (W, b), (gW, gb), (mW, mb, vW, vb) in zip(params, grads,
                                                  state):
        for i in range(len(W)):
            for j in range(len(W[0])):
                mW[i][j] = b1 * mW[i][j] + (1 - b1) * gW[i][j]
                vW[i][j] = b2 * vW[i][j] + (1 - b2) * gW[i][j] ** 2
                W[i][j] -= lr * (mW[i][j] / bc1) / (
                    math.sqrt(vW[i][j] / bc2) + eps)
        for j in range(len(b)):
            mb[j] = b1 * mb[j] + (1 - b1) * gb[j]
            vb[j] = b2 * vb[j] + (1 - b2) * gb[j] ** 2
            b[j] -= lr * (mb[j] / bc1) / (math.sqrt(vb[j] / bc2)
                                          + eps)


# ── one GAN training step (shared by both classes) ───────────────────


def _gan_step(gp, dp, gs, ds, t, real, zg_in, zd_in, lr):
    """One non-saturating GAN update. Returns dl + gl.

    zg_in / zd_in are the full generator INPUTS (noise, or noise ++
    condition), and real is the full discriminator input for the real
    batch (features, or features ++ condition). For the conditional
    case the caller appends the condition to the generator output
    before discriminating; here that is expressed by cond_tail: the
    trailing columns of zg_in past the generator's noise width are
    appended to fakes when the widths differ.
    """
    B = len(real)
    d_in_w = len(dp[0][0])

    def disc_in(feat, gin):
        if len(feat[0]) == d_in_w:
            return feat
        tail = [row[-(d_in_w - len(feat[0])):] for row in gin]
        return [f + tl for f, tl in zip(feat, tail)]

    # ---- discriminator update
    fake, _ = _forward(gp, zd_in)
    dreal = disc_in(real, real)
    dfake = disc_in(fake, zd_in)
    r_out, r_cache = _forward(dp, dreal)
    f_out, f_cache = _forward(dp, dfake)
    dl = 0.0
    dr = []
    for row in r_out:
        s = _sigmoid(row[0])
        dl += -math.log(max(s, 1e-12))
        dr.append([-(1.0 - s) / B])
    df = []
    for row in f_out:
        s = _sigmoid(row[0])
        dl += -math.log(max(1.0 - s, 1e-12))
        df.append([s / B])
    dl /= B
    g_r, _ = _backward(dp, r_cache, dr)
    g_f, _ = _backward(dp, f_cache, df)
    dgrads = [([[a + b for a, b in zip(ra, fa)]
                for ra, fa in zip(rW, fW)],
               [a + b for a, b in zip(rb, fb)])
              for (rW, rb), (fW, fb) in zip(g_r, g_f)]
    _adam_step(dp, dgrads, ds, t, lr)

    # ---- generator update (non-saturating): -log sigmoid(D(G(z)))
    fake, g_cache = _forward(gp, zg_in)
    dfake = disc_in(fake, zg_in)
    f_out, f_cache = _forward(dp, dfake)
    gl = 0.0
    df = []
    for row in f_out:
        s = _sigmoid(row[0])
        gl += -math.log(max(s, 1e-12))
        df.append([-(1.0 - s) / B])
    gl /= B
    dd_in = _input_grad(dp, f_cache, df)
    # gradient flows only into the feature part of the disc input
    nf = len(fake[0])
    d_feat = [row[:nf] for row in dd_in]
    ggrads, _ = _backward(gp, g_cache, d_feat)
    _adam_step(gp, ggrads, gs, t, lr)
    return dl + gl


def _randn(rng, n, m):
    flat = [float(v) for v in rng.normal(0, 1, n * m)._flat()]
    return [flat[i * m:(i + 1) * m] for i in range(n)]


class _Samples(list):
    """(n, k) list-of-rows with the .shape the doctests check."""

    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class SpatialGAN:
    """A small native GAN over 2-D crime/patrol coordinates.

    Parameters
    ----------
    latent_dim : int
        Dimension of the generator's noise input.
    hidden : int
        Width of the hidden layers.
    seed : int
        Seed for parameter initialisation.

    Examples
    --------
    >>> from morie.fn import _array_core as np
    >>> from morie.fairness.gan import SpatialGAN
    >>> rng = np.random.default_rng(0)
    >>> pts = rng.normal([5.0, -3.0], 1.0, size=(800, 2))
    >>> gan = SpatialGAN(seed=0).fit(pts, steps=400)
    >>> samples = gan.sample(500, seed=1)
    >>> samples.shape
    (500, 2)
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64,
                 seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self._gp = None
        self._mean = None
        self._std = None
        self.history: list[float] = []

    def fit(self, points, *, steps: int = 1500, batch_size: int = 128,
            lr: float = 2e-3):
        """Train the GAN on an ``(n, 2)`` array of coordinates."""
        pts = np.asarray(points, dtype=float)
        if len(pts.shape) != 2 or pts.shape[1] != 2:
            raise ValueError("points must be an (n, 2) array")
        rows = pts.tolist()
        n = len(rows)
        if n < 2:
            raise ValueError("need at least two points to fit")

        self._mean = [sum(r[j] for r in rows) / n for j in range(2)]
        self._std = [math.sqrt(sum((r[j] - self._mean[j]) ** 2
                                   for r in rows) / n) + 1e-8
                     for j in range(2)]
        std_pts = [[(r[j] - self._mean[j]) / self._std[j]
                    for j in range(2)] for r in rows]

        rng = np.random.default_rng(self.seed)
        gp = _init_mlp(rng, [self.latent_dim, self.hidden,
                             self.hidden, 2])
        dp = _init_mlp(rng, [2, self.hidden, self.hidden, 1])
        gs, ds = _adam_init(gp), _adam_init(dp)

        bs = min(batch_size, n)
        self.history = []
        for t in range(1, int(steps) + 1):
            idx = [int(v) for v in rng.integers(0, n, bs)._flat()]
            real = [std_pts[i] for i in idx]
            zd = _randn(rng, bs, self.latent_dim)
            zg = _randn(rng, bs, self.latent_dim)
            loss = _gan_step(gp, dp, gs, ds, t, real, zg, zd, lr)
            if t % 50 == 0:
                self.history.append(float(loss))
        self._gp = gp
        return self

    def sample(self, n: int, *, seed: int | None = None):
        """Draw ``n`` synthetic coordinates as an ``(n, 2)`` array."""
        if self._gp is None:
            raise RuntimeError(
                "SpatialGAN is not fitted; call fit() first")
        rng = np.random.default_rng(
            self.seed if seed is None else int(seed))
        z = _randn(rng, int(n), self.latent_dim)
        out, _ = _forward(self._gp, z)
        return _Samples([[v * s + m for v, s, m in
                          zip(row, self._std, self._mean)]
                         for row in out])


class CTGANDebiaser:
    """A conditional tabular GAN that rebalances a biased dataset.

    A native reimplementation of the *debiasing* idea from CTGAN
    (Xu et al., 2019) as used in arXiv:2603.18987 — with no dependency
    on JAX or the Business-Source-licensed ``sdv``/``ctgan`` packages.

    The generator is **conditioned** on two discrete columns — the
    protected ``group`` and the binary ``outcome`` — and learns to
    produce realistic continuous feature columns for each
    ``(group, outcome)`` combination.  Debiasing then works exactly as
    CTGAN's training-by-sampling prescribes: :meth:`debias`
    synthesises a new dataset while sampling the *conditional
    distribution* in a rebalanced way — every group's
    favourable-outcome rate is set to the privileged group's rate — so
    the disparate-impact ratio of the debiased data moves toward 1.

    This redistributes disparity; it does not by itself remove
    structural bias without accompanying policy change.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64,
                 seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self._gp = None
        self._groups = None
        self._feature_cols = None
        self.history: list[float] = []

    def _cond(self, gi, oi):
        """One-hot ``(group, outcome)`` condition rows."""
        ng = len(self._groups)
        out = []
        for g, o in zip(gi, oi):
            row = [0.0] * (ng + 2)
            row[g] = 1.0
            row[ng + o] = 1.0
            out.append(row)
        return out

    def fit(self, df, *, outcome_col, feature_cols,
            group_col="group", favorable=1, steps: int = 1500,
            batch_size: int = 128, lr: float = 2e-3):
        """Train the conditional GAN on a biased DataFrame."""
        feature_cols = list(feature_cols)
        if not feature_cols:
            raise ValueError("need at least one feature column")
        gvals = df[group_col].tolist()
        self._groups = sorted(set(gvals), key=str)
        self._feature_cols = feature_cols
        self._group_col = group_col
        self._outcome_col = outcome_col
        self._favorable = favorable

        g_idx = {g: i for i, g in enumerate(self._groups)}
        gi = [g_idx[g] for g in gvals]
        oi = [1 if v == favorable else 0
              for v in df[outcome_col].tolist()]
        feats = [[float(df[c].tolist()[r]) for c in feature_cols]
                 for r in range(len(gvals))]
        n = len(feats)
        if n < 2:
            raise ValueError("need at least two rows to fit")
        nf = len(feature_cols)

        self._fmean = [sum(f[j] for f in feats) / n
                       for j in range(nf)]
        self._fstd = [math.sqrt(sum((f[j] - self._fmean[j]) ** 2
                                    for f in feats) / n) + 1e-8
                      for j in range(nf)]
        std_feats = [[(f[j] - self._fmean[j]) / self._fstd[j]
                      for j in range(nf)] for f in feats]

        ng = len(self._groups)
        self._group_props = [gi.count(i) / n for i in range(ng)]
        self._group_fav_rate = {
            g: (sum(o for gg, o in zip(gi, oi) if gg == i)
                / max(1, gi.count(i)))
            for i, g in enumerate(self._groups)}
        cond = self._cond(gi, oi)
        cond_dim = ng + 2

        rng = np.random.default_rng(self.seed)
        gp = _init_mlp(rng, [self.latent_dim + cond_dim, self.hidden,
                             self.hidden, nf])
        dp = _init_mlp(rng, [nf + cond_dim, self.hidden, self.hidden,
                             1])
        gs, ds = _adam_init(gp), _adam_init(dp)

        bs = min(batch_size, n)
        self.history = []
        for t in range(1, int(steps) + 1):
            idx = [int(v) for v in rng.integers(0, n, bs)._flat()]
            real = [std_feats[i] + cond[i] for i in idx]
            zg = [zrow + cond[i] for zrow, i in
                  zip(_randn(rng, bs, self.latent_dim), idx)]
            zd = [zrow + cond[i] for zrow, i in
                  zip(_randn(rng, bs, self.latent_dim), idx)]
            loss = _gan_step(gp, dp, gs, ds, t, real, zg, zd, lr)
            if t % 50 == 0:
                self.history.append(float(loss))
        self._gp = gp
        return self

    def debias(self, n: int, *, privileged, seed: int | None = None):
        """Synthesise ``n`` rebalanced rows as a native DataFrame.

        Every group's favourable-outcome rate is set to the privileged
        group's rate, so the disparate-impact ratio of the debiased
        data moves toward 1.
        """
        from morie.fn import _frame_core as pd

        if self._gp is None:
            raise RuntimeError(
                "CTGANDebiaser is not fitted; call fit()")
        if privileged not in self._groups:
            raise ValueError(
                f"privileged group {privileged!r} not seen in "
                f"training; groups: {self._groups}")
        target_rate = self._group_fav_rate[privileged]
        rng = np.random.default_rng(
            self.seed if seed is None else int(seed))
        ng = len(self._groups)
        cum = []
        acc = 0.0
        for p in self._group_props:
            acc += p
            cum.append(acc)
        gi = []
        for u in rng.uniform(0, 1, int(n))._flat():
            for i, c in enumerate(cum):
                if float(u) <= c or i == ng - 1:
                    gi.append(i)
                    break
        oi = [1 if float(u) < target_rate else 0
              for u in rng.uniform(0, 1, int(n))._flat()]
        cond = self._cond(gi, oi)
        z = [zrow + crow for zrow, crow in
             zip(_randn(rng, int(n), self.latent_dim), cond)]
        std_feat, _ = _forward(self._gp, z)
        out = {
            self._group_col: [self._groups[i] for i in gi],
            self._outcome_col: [self._favorable if o == 1 else 0
                                for o in oi],
        }
        for j, col in enumerate(self._feature_cols):
            out[col] = [row[j] * self._fstd[j] + self._fmean[j]
                        for row in std_feat]
        return pd.DataFrame(out)
