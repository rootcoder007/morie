# morie.fn -- function file (rootcoder007/morie)
"""Average output noise power."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch4_average_output_noise_power"]


def rangayyan_ch4_average_output_noise_power(P_eta_i, H, freqs=None, df=None):
    r"""Average output noise power of a filter (Rangayyan Ch. 4):

    .. math:: P_{\eta_o} = \frac{P_{\eta_i}}{2}
              \int_{-\infty}^{\infty} |H(f)|^2\, df.

    White input noise is shaped by the filter's energy, so the output
    power depends only on :math:`\int |H|^2` -- the noise-equivalent
    bandwidth. A filter with unit passband gain still amplifies noise
    in proportion to how wide it is, which is why narrowing the band
    is the primary noise-reduction lever.

    Parameters
    ----------
    P_eta_i : float
        Input noise power spectral density (two-sided).
    H : array-like
        Filter frequency response (complex or magnitude).
    freqs : array-like, optional
        Matching frequencies, for the integration measure.
    df : float, optional
        Uniform frequency spacing when freqs is omitted.

    Returns
    -------
    RichResult
        keys: ``output_power``, ``energy_integral``,
        ``noise_equivalent_bw``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4 (noise through filters).
    """
    Hm = np.abs(np.asarray(H, dtype=complex).ravel()) ** 2
    if Hm.size < 2:
        raise ValueError("H must have at least 2 points.")
    P_in = float(P_eta_i)
    if P_in < 0:
        raise ValueError("input noise power cannot be negative.")
    if freqs is not None:
        f = np.asarray(freqs, dtype=float).ravel()
        if f.size != Hm.size:
            raise ValueError("freqs must match the length of H.")
        integral = float(np.trapezoid(Hm, f))
    else:
        step = 1.0 if df is None else float(df)
        if step <= 0:
            raise ValueError("df must be positive.")
        integral = float(np.trapezoid(Hm, dx=step))
    peak = float(Hm.max())
    return RichResult(payload={"output_power": P_in / 2.0 * integral,
                               "energy_integral": integral,
                               "noise_equivalent_bw": integral / peak if peak > 0 else 0.0,
                               "method": "P_out = (P_in/2) int |H(f)|^2 df"})


def cheatsheet():
    return "rng211: output noise tracks int|H|^2 -- narrower band, less noise"
