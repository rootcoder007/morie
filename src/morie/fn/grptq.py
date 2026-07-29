# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static post-training quantization: calibrate activation scales on a sample."""

import numpy as np

from ._richresult import RichResult
from .grq8 import quantize_symmetric

__all__ = ["geron_static_ptq"]

_METHOD = "Static post-training quantization (activation calibration)"


def geron_static_ptq(model, calibration_data, bits=8, percentile=100.0):
    r"""Run a representative batch to fix each layer's activation scale.

    .. math::
        s_{\text{act}} = \frac{\mathrm{calib}(|a|)}{q_{\max}},
        \qquad x_q = \mathrm{round}(x / s_{\text{act}})

    Weights can be quantized offline because they are known; activations
    cannot, so *static* PTQ borrows a calibration set to stand in for
    them.  That is the entire method, and its failure mode follows
    directly: a calibration set that misses the real input range gives a
    scale that clips in production.  ``percentile`` below 100 trades a
    little clipping for resolution, which is what saves you when one
    outlier activation would otherwise set the scale for the whole
    tensor.

    ``model`` is caller-supplied: a sequence of callables, each mapping
    an activation array to the next.  The contract is enforced -- every
    layer must return a finite array with the same leading (batch) size.

    Parameters
    ----------
    model : sequence of callables
        ``layer(activations) -> activations``.
    calibration_data : array-like, shape (n_samples, ...)
        Representative inputs; must not be empty.
    bits : int, optional
    percentile : float, optional
        Calibration percentile of ``|activation|`` in ``(0, 100]``.

    Returns
    -------
    RichResult
        Payload keys ``scales`` (per layer, input included),
        ``activation_ranges``, ``quantized_output``, ``output``,
        ``max_abs_error``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Appendix B, Static Quantization (PTQ) section.

    Examples
    --------
    One layer doubling its input, calibration values up to 2, so the
    output range is 4 and the output scale is ``4/127``:

    >>> r = geron_static_ptq([lambda a: 2 * a], [[1.0], [2.0]])
    >>> [round(v, 6) for v in r["activation_ranges"]]
    [2.0, 4.0]
    >>> round(r["scales"][1], 8)
    0.03149606

    The quantized output stays within half a step of the float one:

    >>> r["max_abs_error"] <= r["scales"][-1] / 2 + 1e-12
    True
    """
    layers = list(model)
    if not layers:
        raise ValueError("model has no layers; nothing to calibrate.")
    for i, f in enumerate(layers):
        if not callable(f):
            raise ValueError(f"model[{i}] is not callable ({type(f).__name__}).")
    A = np.asarray(calibration_data, dtype=float)
    if A.size == 0:
        raise ValueError("calibration_data is empty; static PTQ needs a representative sample.")
    if not np.all(np.isfinite(A)):
        raise ValueError("calibration_data contains non-finite values.")
    percentile = float(percentile)
    if not (0.0 < percentile <= 100.0):
        raise ValueError(f"percentile must lie in (0, 100], got {percentile}.")

    def _range(arr):
        return float(np.percentile(np.abs(arr), percentile))

    acts = A
    ranges = [_range(acts)]
    outputs = [acts]
    for i, f in enumerate(layers):
        nxt = np.asarray(f(acts), dtype=float)
        if not np.all(np.isfinite(nxt)):
            raise ValueError(f"model[{i}] returned non-finite activations.")
        if nxt.shape[0] != acts.shape[0]:
            raise ValueError(
                f"model[{i}] changed the batch size from {acts.shape[0]} to {nxt.shape[0]}."
            )
        acts = nxt
        ranges.append(_range(acts))
        outputs.append(acts)

    qmax = 2 ** (int(bits) - 1) - 1
    scales = []
    for r in ranges:
        if r == 0:
            raise ValueError(
                "an activation tensor is identically zero on the calibration set; "
                "its scale would be zero."
            )
        scales.append(r / qmax)

    q_out, s_out, deq = quantize_symmetric(outputs[-1], bits)
    err = float(np.max(np.abs(deq - outputs[-1])))

    return RichResult(
        title="Static post-training quantization",
        summary_lines=[("Layers", len(layers)), ("Output scale", float(scales[-1]))],
        payload={
            "scales": scales,
            "activation_ranges": ranges,
            "quantized_output": q_out.tolist(),
            "dequantized_output": deq.tolist(),
            "output": outputs[-1].tolist(),
            "max_abs_error": err,
            "percentile": percentile,
            "estimate": scales,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grptq: run calibration batch through the layers, s_act = calib(|a|)/qmax per tensor"
