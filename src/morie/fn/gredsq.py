# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoder-decoder (seq2seq) decoding loop."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_encoder_decoder_seq2seq"]

_METHOD = "Encoder-decoder seq2seq decoding loop"


def geron_encoder_decoder_seq2seq(encoder, decoder, x, max_out_len,
                                  start_token=None, eos_token=None):
    r"""Encode once, then decode step by step.

    .. math::
        c = \mathrm{encoder}(x_{1:T}),\qquad
        y_t = \mathrm{decoder}(y_{t-1}, c, s_t)

    The whole input is compressed into one context vector ``c`` before
    a single output token is produced -- which is what makes the
    architecture work for reordering languages, and equally what makes
    it the bottleneck attention was invented to remove: ``c`` is the
    same size whether the source has five words or five hundred.

    The models are the caller's. Contracts, enforced here:

    * ``encoder(x) -> c``, a finite 1-D array;
    * ``decoder(y_prev, c, t) -> y_t``, finite, and the same shape at
      every step.

    Decoding is greedy and stops at ``max_out_len`` or when ``y_t``
    equals ``eos_token``.

    Parameters
    ----------
    encoder, decoder : callable
        See contracts above.
    x : array-like
        Input sequence.
    max_out_len : int
        Hard cap on generated steps, at least 1.
    start_token : array-like or scalar, optional
        ``y_0``; defaults to zeros shaped like ``c``.
    eos_token : scalar, optional
        Stop symbol, compared against ``y_t`` when the output is scalar.

    Returns
    -------
    RichResult
        Payload keys ``outputs``, ``context``, ``n_steps``,
        ``stopped_early``, ``context_dim``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 14, Encoder-Decoder / Seq2Seq section.

    Examples
    --------
    A toy pair: the encoder sums the input, the decoder counts down from
    it. The context is computed once and reused at every step.

    >>> enc = lambda x: [float(sum(x))]
    >>> dec = lambda y_prev, c, t: c[0] - t
    >>> r = geron_encoder_decoder_seq2seq(enc, dec, [1.0, 2.0], max_out_len=3)
    >>> r["context"]
    [3.0]
    >>> r["outputs"]
    [2.0, 1.0, 0.0]

    An ``eos_token`` cuts the loop short:

    >>> r2 = geron_encoder_decoder_seq2seq(enc, dec, [1.0, 2.0], max_out_len=10,
    ...                                    eos_token=0.0)
    >>> r2["outputs"], r2["n_steps"], r2["stopped_early"]
    ([2.0, 1.0, 0.0], 3, True)

    A decoder that returns garbage is caught rather than propagated:

    >>> bad = lambda y_prev, c, t: float("nan")
    >>> geron_encoder_decoder_seq2seq(enc, bad, [1.0], max_out_len=2)
    Traceback (most recent call last):
        ...
    ValueError: decoder returned a non-finite value at step 1.
    """
    if not callable(encoder):
        raise ValueError(f"encoder must be callable, got {type(encoder).__name__}.")
    if not callable(decoder):
        raise ValueError(f"decoder must be callable, got {type(decoder).__name__}.")
    max_out_len = int(max_out_len)
    if max_out_len < 1:
        raise ValueError(f"max_out_len must be at least 1, got {max_out_len}.")

    c = np.asarray(encoder(x), dtype=float).ravel()
    if c.size == 0:
        raise ValueError("encoder returned an empty context vector.")
    if not np.all(np.isfinite(c)):
        raise ValueError("encoder returned a non-finite context vector.")

    y_prev = np.zeros_like(c) if start_token is None else np.asarray(start_token, dtype=float)
    outputs = []
    stopped = False
    first_shape = None
    for t in range(1, max_out_len + 1):
        y = np.asarray(decoder(y_prev, c, t), dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"decoder returned a non-finite value at step {t}.")
        if first_shape is None:
            first_shape = y.shape
        elif y.shape != first_shape:
            raise ValueError(
                f"decoder changed output shape at step {t}: {y.shape} after {first_shape}."
            )
        outputs.append(float(y) if y.ndim == 0 else y.tolist())
        y_prev = y
        if eos_token is not None and y.size == 1 and float(y) == float(eos_token):
            stopped = True
            break

    return RichResult(
        title="Encoder-decoder seq2seq",
        summary_lines=[("Context dim", int(c.size)), ("Steps", len(outputs)),
                       ("Stopped at EOS", stopped)],
        payload={
            "outputs": outputs,
            "context": c.tolist(),
            "context_dim": int(c.size),
            "n_steps": len(outputs),
            "stopped_early": bool(stopped),
            "estimate": outputs,
            "n": len(outputs),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "gredsq: c = encoder(x) once, then greedy y_t = decoder(y_{t-1}, c, t) with enforced contracts"
