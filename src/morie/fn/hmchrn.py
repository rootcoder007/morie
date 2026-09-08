# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Character-level RNN language model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_char_rnn"]


def geron_char_rnn(text, hidden=8, epochs=50, lr=0.1, seed=0, generate=0):
    """
    Character-level RNN language model.

    Formula: P(c_t | c_{<t}) = softmax(W h_t)

    A vanilla RNN trained by real backpropagation through time on the
    character sequence:

        h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h),
        logits_t = W_hy h_t + b_y,

    with cross-entropy against the next character. The output weights
    start at exactly zero, so the model begins uniform and the first
    reported loss is exactly ``log V`` -- a computed baseline, not an
    approximation, which makes "is it learning?" answerable by comparison
    with ``chance_loss``.

    Recurrent and input weights come from a deterministic LCG draw, so a
    run reproduces. BPTT is over the whole string (no truncation), which
    is the honest version for short texts and where vanishing gradients
    would show up for long ones.

    ``generate`` continues the text greedily from the last character,
    which is the cheapest possible check that the model learned any
    structure at all.

    Parameters
    ----------
    text : str or sequence
        Training sequence; needs at least 2 characters and 2 distinct
        symbols.
    hidden : int, default 8
        Hidden state width.
    epochs : int, default 50
    lr : float, default 0.1
    seed : int, default 0
    generate : int, default 0
        Characters to generate greedily after training.

    Returns
    -------
    result : RichResult
        Keys: loss_history, chance_loss, perplexity, vocab, vocab_size,
        weights, generated, final_loss, estimate, n, method.

    Examples
    --------
    A repeating pattern over a 2-symbol vocabulary: the initial loss is
    exactly ``log 2`` and training drives it down.

    >>> import math
    >>> r = geron_char_rnn("ababababab", hidden=6, epochs=200, lr=0.5, seed=1)
    >>> round(r["chance_loss"], 9) == round(math.log(2), 9)
    True
    >>> round(r["loss_history"][0], 9) == round(math.log(2), 9)
    True
    >>> r["final_loss"] < 0.05
    True
    >>> r["vocab"], r["vocab_size"]
    (['a', 'b'], 2)

    Perplexity is the exponential of the loss, so a learned alternation
    scores well under the 2.0 of a coin flip:

    >>> r["perplexity"] < 1.1
    True

    Greedy continuation of "abab..." keeps alternating:

    >>> r2 = geron_char_rnn("ababababab", hidden=6, epochs=300, lr=0.5, seed=1, generate=4)
    >>> r2["generated"]
    'abab'

    References
    ----------
    Géron Ch 14
    """
    chars = list(text)
    if len(chars) < 2:
        raise ValueError(f"geron_char_rnn: text needs at least 2 characters, got {len(chars)}")
    vocab = sorted(set(chars), key=str)
    V = len(vocab)
    if V < 2:
        raise ValueError("geron_char_rnn: text uses a single symbol; a language model over it is trivial")
    index = {c: i for i, c in enumerate(vocab)}
    seq = np.array([index[c] for c in chars], dtype=int)
    H = int(hidden)
    if H < 1:
        raise ValueError(f"geron_char_rnn: hidden must be >= 1, got {hidden!r}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_char_rnn: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if eta < 0:
        raise ValueError(f"geron_char_rnn: lr must be non-negative, got {lr!r}")
    G = int(generate)
    if G < 0:
        raise ValueError(f"geron_char_rnn: generate must be non-negative, got {generate!r}")

    s = int(seed) % 2**32

    def draw(shape, sd):
        nonlocal s
        n = int(np.prod(shape))
        u = np.empty(n)
        for i in range(n):
            s = (1664525 * s + 1013904223) % 2**32
            u[i] = (s + 0.5) / 2**32
        return ((2.0 * u - 1.0) * np.sqrt(3.0) * sd).reshape(shape)

    Wxh = draw((V, H), 1.0 / np.sqrt(V))
    Whh = draw((H, H), 1.0 / np.sqrt(H))
    bh = np.zeros(H)
    Why = np.zeros((H, V))
    by = np.zeros(V)

    T = seq.size - 1
    hist = []
    for _ in range(E):
        hs = [np.zeros(H)]
        ps = []
        loss = 0.0
        for t in range(T):
            x = np.zeros(V)
            x[seq[t]] = 1.0
            h = np.tanh(x @ Wxh + hs[-1] @ Whh + bh)
            hs.append(h)
            z = h @ Why + by
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            ps.append(p)
            loss += -float(np.log(p[seq[t + 1]]))
        hist.append(loss / T)

        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dbh = np.zeros_like(bh)
        dWhy = np.zeros_like(Why)
        dby = np.zeros_like(by)
        dh_next = np.zeros(H)
        for t in range(T - 1, -1, -1):
            dz = ps[t].copy()
            dz[seq[t + 1]] -= 1.0
            dz /= T
            dWhy += np.outer(hs[t + 1], dz)
            dby += dz
            dh = Why @ dz + dh_next
            draw_ = (1.0 - hs[t + 1] ** 2) * dh
            x = np.zeros(V)
            x[seq[t]] = 1.0
            dWxh += np.outer(x, draw_)
            dWhh += np.outer(hs[t], draw_)
            dbh += draw_
            dh_next = Whh @ draw_
        for arr, g in ((Wxh, dWxh), (Whh, dWhh), (bh, dbh), (Why, dWhy), (by, dby)):
            arr -= eta * g

    # Final loss and greedy generation.
    h = np.zeros(H)
    loss = 0.0
    for t in range(T):
        x = np.zeros(V)
        x[seq[t]] = 1.0
        h = np.tanh(x @ Wxh + h @ Whh + bh)
        z = h @ Why + by
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        loss += -float(np.log(p[seq[t + 1]]))
    final = loss / T

    gen = ""
    if G:
        gh = np.zeros(H)
        cur = seq[0]
        for t in range(T + 1):
            x = np.zeros(V)
            x[cur] = 1.0
            gh = np.tanh(x @ Wxh + gh @ Whh + bh)
            cur = seq[t + 1] if t < T else int(np.argmax(gh @ Why + by))
        for _ in range(G):
            gen += vocab[cur]
            x = np.zeros(V)
            x[cur] = 1.0
            gh = np.tanh(x @ Wxh + gh @ Whh + bh)
            cur = int(np.argmax(gh @ Why + by))

    return RichResult(
        title="Character-level RNN",
        summary_lines=[("Vocabulary", V), ("Final loss", final), ("Perplexity", float(np.exp(final)))],
        interpretation="A uniform model scores log V; anything below that is structure the RNN has captured.",
        payload={
            "loss_history": hist,
            "final_loss": final,
            "chance_loss": float(np.log(V)),
            "perplexity": float(np.exp(final)),
            "vocab": vocab,
            "vocab_size": int(V),
            "weights": {"Wxh": Wxh.tolist(), "Whh": Whh.tolist(), "bh": bh.tolist(), "Why": Why.tolist(), "by": by.tolist()},
            "hidden": H,
            "generated": gen,
            "estimate": final,
            "n": int(T),
            "method": "vanilla char-RNN trained with full backpropagation through time",
        },
    )


def cheatsheet():
    return "hmchrn: Character-level RNN language model"


# compact alias per ledger/NAMING.md
geroncharrnn = geron_char_rnn
