"""Morphological opening of a 2-D image with a flat structuring element.

Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press, ch. 2 and 12.
Matheron, G. (1975). Random Sets and Integral Geometry. Wiley.
"""

from ._richresult import RichResult

__all__ = ["morphological_opening"]


def _pass(img, se, oy, ox, op):
    h, w = len(img), len(img[0])
    out = [[0.0] * w for _ in range(h)]
    offs = [(dy - oy, dx - ox) for dy in range(len(se)) for dx in range(len(se[0])) if se[dy][dx]]
    for y in range(h):
        for x in range(w):
            vals = []
            for dy, dx in offs:
                if op == "erode":
                    yy, xx = y + dy, x + dx
                else:
                    yy, xx = y - dy, x - dx
                if 0 <= yy < h and 0 <= xx < w:
                    vals.append(img[yy][xx])
            out[y][x] = (min(vals) if op == "erode" else max(vals)) if vals else img[y][x]
    return out


def morphological_opening(image, structure=None, origin=None):
    r"""Opening = dilation of the erosion: (f erode B) dilate B.

    Grey-scale erosion (f erode B)(x) = min_{b in B} f(x + b) and dilation
    (f dilate B)(x) = max_{b in B} f(x - b) with a flat structuring element B; positions
    outside the image are ignored (+inf for erosion, -inf for dilation, as in Soille 2003). Binary images
    (0/1) give the binary opening. The result is anti-extensive (<= f), increasing and
    idempotent (Serra 1982).

    Parameters
    ----------
    image : 2-D list of numbers
    structure : 2-D list of 0/1, optional
        Default: the 3 x 3 square.
    origin : (row, col), optional
        Default: the centre of the structuring element.

    Returns
    -------
    RichResult
        Keys: opened, eroded.

    References
    ----------
    Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.
    Soille, P. (2003). Morphological Image Analysis, 2nd ed. Springer.

    Examples
    --------
    >>> img = [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    >>> morphological_opening(img, [[1, 1], [1, 1]], (0, 0))["opened"][3]
    [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    img = [[float(v) for v in r] for r in image]
    if not img or not img[0] or any(len(r) != len(img[0]) for r in img):
        raise ValueError("image must be a non-empty rectangular 2-D array")
    se = [[1, 1, 1]] * 3 if structure is None else [[1 if v else 0 for v in r] for r in structure]
    if not any(any(r) for r in se):
        raise ValueError("structuring element is empty")
    oy, ox = (len(se) // 2, len(se[0]) // 2) if origin is None else (int(origin[0]), int(origin[1]))
    er = _pass(img, se, oy, ox, "erode")
    op = _pass(er, se, oy, ox, "dilate")
    return RichResult(
        title="Morphological opening",
        summary_lines=[("pixels", len(img) * len(img[0]))],
        payload={"opened": op, "eroded": er},
    )


def cheatsheet():
    return "mopnop: grey/binary morphological opening with a flat structuring element"
