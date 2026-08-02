# morie.fn -- native plotting core (rootcoder007/morie)
"""Native plotting: the pyplot surface morie uses, with no matplotlib.

Figures are recorded as a vector scene graph and rendered to SVG
directly or rasterized to PNG through a scanline renderer + the stdlib
zlib PNG encoder (PNG spec: RFC 2083; filter type 0, 8-bit RGB).
Text uses an embedded 5x7 bitmap font (classic public-domain glyph set)
scaled by font size.

Covered surface (from an AST survey of every call site in morie):
subplots, plot, scatter, bar, barh, hist, imshow, pcolormesh,
fill_between, axhline, axvline, text, annotate, legend, grid,
set_title/xlabel/ylabel/xlim/ylim/xticks/yticks/xticklabels/
yticklabels, suptitle, tight_layout, colorbar (no-op layout),
savefig (.svg native vector, .png rasterized), close.

Not a general matplotlib replacement: it renders what morie draws.
Unknown methods raise AttributeError so gaps surface loudly.
"""

from __future__ import annotations

import math
import struct
import zlib

# ------------------------------------------------------------- font
# 5x7 bitmap font, one glyph per printable ASCII char (32-126).
# Each glyph: 5 column bytes, LSB = top row (the classic layout used
# by countless character LCDs; public domain).
_FONT = {}


def _deffont():
    raw = {
        " ": "0000000000", "!": "00005F0000", '"': "0007000700",
        "#": "147F147F14", "$": "242A7F2A12", "%": "2313086462",
        "&": "3649552250", "'": "0005030000", "(": "001C224100",
        ")": "0041221C00", "*": "2A1C7F1C2A", "+": "08083E0808",
        ",": "0050300000", "-": "0808080808", ".": "0060600000",
        "/": "2010080402", "0": "3E51494538", "1": "00427F4000",
        "2": "4261514946", "3": "2141454B31", "4": "1814127F10",
        "5": "2745454539", "6": "3C4A494930", "7": "0171090503",
        "8": "3649494936", "9": "064949291E", ":": "0036360000",
        ";": "0056360000", "<": "0814224100", "=": "1414141414",
        ">": "0041221408", "?": "0201510906", "@": "324979413E",
        "A": "7E1111117E", "B": "7F49494936", "C": "3E41414122",
        "D": "7F4141221C", "E": "7F49494941", "F": "7F09090901",
        "G": "3E41494A7A", "H": "7F0808087F", "I": "00417F4100",
        "J": "2040413F01", "K": "7F08142241", "L": "7F40404040",
        "M": "7F020C027F", "N": "7F0408107F", "O": "3E4141413E",
        "P": "7F09090906", "Q": "3E4151215E", "R": "7F09192946",
        "S": "4649494931", "T": "01017F0101", "U": "3F4040403F",
        "V": "1F2040201F", "W": "3F4038403F", "X": "6314081463",
        "Y": "0708700807", "Z": "6151494543", "[": "007F414100",
        "\\\\": "0204081020", "]": "0041417F00", "^": "0402010204",
        "_": "4040404040", "`": "0001020400", "a": "2054545478",
        "b": "7F48444438", "c": "3844444420", "d": "384444487F",
        "e": "3854545418", "f": "087E090102", "g": "0C5252523E",
        "h": "7F08040478", "i": "00447D4000", "j": "2040443D00",
        "k": "7F10284400", "l": "00417F4000", "m": "7C04180478",
        "n": "7C08040478", "o": "3844444438", "p": "7C14141408",
        "q": "0814141878", "r": "7C08040408", "s": "4854545420",
        "t": "043F444020", "u": "3C4040207C", "v": "1C2040201C",
        "w": "3C4030403C", "x": "4428102844", "y": "0C5050503C",
        "z": "4464544C44", "{": "0008364100", "|": "00007F0000",
        "}": "0041360800", "~": "0808041008",
    }
    for ch, hexs in raw.items():
        key = "\\" if ch == "\\\\" else ch
        _FONT[key] = [int(hexs[i:i + 2], 16) for i in range(0, 10, 2)]


_deffont()

_COLORS = {
    "b": (31, 119, 180), "blue": (31, 119, 180),
    "g": (44, 160, 44), "green": (44, 160, 44),
    "r": (214, 39, 40), "red": (214, 39, 40),
    "c": (23, 190, 207), "cyan": (23, 190, 207),
    "m": (227, 119, 194), "magenta": (227, 119, 194),
    "y": (188, 189, 34), "yellow": (188, 189, 34),
    "k": (0, 0, 0), "black": (0, 0, 0),
    "w": (255, 255, 255), "white": (255, 255, 255),
    "orange": (255, 127, 14), "purple": (148, 103, 189),
    "gray": (127, 127, 127), "grey": (127, 127, 127),
    "steelblue": (70, 130, 180), "tab:blue": (31, 119, 180),
    "tab:orange": (255, 127, 14), "tab:green": (44, 160, 44),
    "tab:red": (214, 39, 40), "tab:purple": (148, 103, 189),
    "tab:gray": (127, 127, 127), "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211), "darkred": (139, 0, 0),
    "navy": (0, 0, 128), "teal": (0, 128, 128),
    "gold": (255, 215, 0), "crimson": (220, 20, 60),
}
_CYCLE = ["tab:blue", "tab:orange", "tab:green", "tab:red",
          "tab:purple", "gray", "crimson", "teal"]


def _rgb(c, default=(31, 119, 180)):
    if c is None:
        return default
    if isinstance(c, (tuple, list)):
        v = list(c)[:3]
        return tuple(int(round(255 * x)) if x <= 1 else int(x)
                     for x in v)
    c = str(c)
    if c.startswith("#") and len(c) >= 7:
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
    if c.startswith("C") and c[1:].isdigit():
        return _COLORS[_CYCLE[int(c[1:]) % len(_CYCLE)]]
    return _COLORS.get(c.lower(), default)


def _flt(v):
    if hasattr(v, "_flat"):
        return [float(x) for x in v._flat()]
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in v]


def _viridis(t):
    """Small piecewise-linear approximation of the viridis ramp."""
    stops = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)),
             (0.5, (33, 145, 140)), (0.75, (94, 201, 98)),
             (1.0, (253, 231, 37))]
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(int(round(a + f * (b - a)))
                         for a, b in zip(c0, c1))
    return stops[-1][1]


def get_cmap(name="viridis"):
    del name
    return _viridis


# ------------------------------------------------------------ scene
class Axes:
    def __init__(self, fig, rect):
        self.fig = fig
        self.rect = rect            # (x0, y0, w, h) figure fractions
        self.items = []             # draw commands in data space
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self._xlim = None
        self._ylim = None
        self._xticks = None
        self._yticks = None
        self._xticklabels = None
        self._yticklabels = None
        self._legend = []
        self._show_legend = False
        self._grid = False
        self._ci = 0
        self._invert_y = False

    # ---------------------------------------------------- plotting
    def _next_color(self):
        c = _CYCLE[self._ci % len(_CYCLE)]
        self._ci += 1
        return c

    def plot(self, x, y=None, fmt=None, *, color=None, label=None,
             linewidth=1.5, lw=None, linestyle="-", ls=None,
             marker=None, alpha=1.0, **kw):
        del kw
        if y is None:
            y = x
            x = list(range(len(_flt(y))))
        xs, ys = _flt(x), _flt(y)
        style = ls or linestyle
        if isinstance(fmt, str):
            for ch, name in (("o", "o"), ("s", "s"), ("^", "^")):
                if ch in fmt:
                    marker = name
            if "--" in fmt:
                style = "--"
            elif ":" in fmt:
                style = ":"
            for ch in fmt:
                if ch in _COLORS:
                    color = ch
        col = _rgb(color) if color is not None else \
            _rgb(self._next_color())
        self.items.append(("line", xs, ys, col, lw or linewidth,
                           style, alpha))
        if marker:
            self.items.append(("scatter", xs, ys, [col] * len(xs),
                               [16.0] * len(xs), alpha))
        if label:
            self._legend.append((label, col))
        return [None]

    def scatter(self, x, y, *, s=20, c=None, color=None, label=None,
                alpha=1.0, cmap=None, marker=None, edgecolors=None,
                **kw):
        del kw, marker, edgecolors
        xs, ys = _flt(x), _flt(y)
        base = c if c is not None else color
        if base is not None and not isinstance(base, str) and \
                hasattr(base, "__len__") and len(_flt_safe(base)) == len(xs):
            vals = _flt(base)
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            ramp = cmap if callable(cmap) else _viridis
            cols = [ramp((v - lo) / span) for v in vals]
        else:
            col = _rgb(base) if base is not None else \
                _rgb(self._next_color())
            cols = [col] * len(xs)
        try:
            sizes = _flt(s)
            if len(sizes) != len(xs):
                sizes = [sizes[0]] * len(xs)
        except Exception:
            sizes = [20.0] * len(xs)
        self.items.append(("scatter", xs, ys, cols, sizes, alpha))
        if label:
            self._legend.append((label, cols[0]))
        return None

    def bar(self, x, height, *, width=0.8, color=None, label=None,
            alpha=1.0, bottom=None, **kw):
        del kw
        xs, hs = _flt(x), _flt(height)
        bo = _flt(bottom) if bottom is not None else [0.0] * len(xs)
        col = _rgb(color) if color is not None else \
            _rgb(self._next_color())
        self.items.append(("bar", xs, hs, bo, float(width), col,
                           alpha, False))
        if label:
            self._legend.append((label, col))
        return None

    def barh(self, y, width, *, height=0.8, color=None, label=None,
             alpha=1.0, **kw):
        del kw
        ys, ws = _flt(y), _flt(width)
        col = _rgb(color) if color is not None else \
            _rgb(self._next_color())
        self.items.append(("bar", ys, ws, [0.0] * len(ys),
                           float(height), col, alpha, True))
        if label:
            self._legend.append((label, col))
        return None

    def hist(self, x, bins=10, *, color=None, alpha=1.0, label=None,
             density=False, **kw):
        del kw
        vals = sorted(_flt(x))
        if not vals:
            return None, None, None
        if isinstance(bins, int):
            lo, hi = vals[0], vals[-1]
            span = (hi - lo) or 1.0
            edges = [lo + span * i / bins for i in range(bins + 1)]
        else:
            edges = _flt(bins)
        counts = [0.0] * (len(edges) - 1)
        for v in vals:
            for j in range(len(counts)):
                if edges[j] <= v <= edges[j + 1] and \
                        (v < edges[j + 1] or j == len(counts) - 1):
                    counts[j] += 1
                    break
        if density:
            n = len(vals)
            counts = [c / (n * (edges[j + 1] - edges[j]) or 1.0)
                      for j, c in enumerate(counts)]
        centers = [(edges[j] + edges[j + 1]) / 2
                   for j in range(len(counts))]
        w = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
        col = _rgb(color) if color is not None else \
            _rgb(self._next_color())
        self.items.append(("bar", centers, counts,
                           [0.0] * len(counts), w, col, alpha, False))
        if label:
            self._legend.append((label, col))
        return counts, edges, None

    def imshow(self, img, *, cmap=None, aspect=None, origin=None,
               vmin=None, vmax=None, extent=None, alpha=1.0,
               interpolation=None, **kw):
        del kw, aspect, interpolation
        rows = img.tolist() if hasattr(img, "tolist") else \
            [list(r) for r in img]
        rows = [[float(v) for v in r] for r in rows]
        flat = [v for r in rows for v in r]
        lo = vmin if vmin is not None else min(flat)
        hi = vmax if vmax is not None else max(flat)
        span = (hi - lo) or 1.0
        ramp = cmap if callable(cmap) else _viridis
        grid = [[ramp((v - lo) / span) for v in r] for r in rows]
        if origin != "lower":
            grid = grid[::-1]
        self.items.append(("mesh", grid, extent, alpha))
        return None

    def pcolormesh(self, *args, cmap=None, alpha=1.0, **kw):
        del kw
        C = args[-1]
        return self.imshow(C, cmap=cmap, alpha=alpha, origin="lower")

    def fill_between(self, x, y1, y2=0.0, *, color=None, alpha=0.3,
                     label=None, **kw):
        del kw
        xs = _flt(x)
        a = _flt(y1)
        b = _flt(y2) if hasattr(y2, "__len__") or \
            not isinstance(y2, (int, float)) else [float(y2)] * len(xs)
        col = _rgb(color) if color is not None else \
            _rgb(self._next_color())
        self.items.append(("fill", xs, a, b, col, alpha))
        if label:
            self._legend.append((label, col))
        return None

    def axhline(self, y=0.0, *, color="k", linestyle="--", ls=None,
                linewidth=1.0, lw=None, alpha=1.0, label=None, **kw):
        del kw
        self.items.append(("hline", float(y), _rgb(color),
                           lw or linewidth, ls or linestyle, alpha))
        if label:
            self._legend.append((label, _rgb(color)))
        return None

    def axvline(self, x=0.0, *, color="k", linestyle="--", ls=None,
                linewidth=1.0, lw=None, alpha=1.0, label=None, **kw):
        del kw
        self.items.append(("vline", float(x), _rgb(color),
                           lw or linewidth, ls or linestyle, alpha))
        if label:
            self._legend.append((label, _rgb(color)))
        return None

    def text(self, x, y, s, *, fontsize=10, color="k", ha="left",
             va="baseline", rotation=0, **kw):
        del kw, rotation
        self.items.append(("text", float(x), float(y), str(s),
                           float(fontsize), _rgb(color), ha, va))
        return None

    def annotate(self, s, xy, xytext=None, **kw):
        del kw
        pos = xytext if xytext is not None else xy
        return self.text(float(pos[0]), float(pos[1]), s)

    def violinplot(self, dataset, positions=None, **kw):
        del kw
        cols = dataset if isinstance(dataset[0], (list, tuple)) or \
            hasattr(dataset[0], "__len__") else [dataset]
        pos = positions or list(range(1, len(cols) + 1))
        for p, col in zip(pos, cols):
            vals = sorted(_flt(col))
            if not vals:
                continue
            q1 = vals[len(vals) // 4]
            q3 = vals[(3 * len(vals)) // 4]
            self.items.append(("bar", [p], [q3 - q1], [q1], 0.5,
                               _rgb("tab:blue"), 0.5, False))
            med = vals[len(vals) // 2]
            self.items.append(("scatter", [p], [med],
                               [_rgb("k")], [16.0], 1.0))
        return {}

    # -------------------------------------------------- decoration
    def set_title(self, s, **kw):
        del kw
        self.title = str(s)

    def set_xlabel(self, s, **kw):
        del kw
        self.xlabel = str(s)

    def set_ylabel(self, s, **kw):
        del kw
        self.ylabel = str(s)

    def set_xlim(self, a=None, b=None, **kw):
        del kw
        if isinstance(a, (tuple, list)):
            a, b = a
        self._xlim = (float(a), float(b))

    def set_ylim(self, a=None, b=None, **kw):
        del kw
        if isinstance(a, (tuple, list)):
            a, b = a
        self._ylim = (float(a), float(b))

    def set_xticks(self, t, labels=None, **kw):
        del kw
        self._xticks = _flt(t)
        if labels is not None:
            self._xticklabels = [str(v) for v in labels]

    def set_yticks(self, t, labels=None, **kw):
        del kw
        self._yticks = _flt(t)
        if labels is not None:
            self._yticklabels = [str(v) for v in labels]

    def set_xticklabels(self, labels, **kw):
        del kw
        self._xticklabels = [str(v) for v in labels]

    def set_yticklabels(self, labels, **kw):
        del kw
        self._yticklabels = [str(v) for v in labels]

    def legend(self, *a, **kw):
        del a, kw
        self._show_legend = True

    def grid(self, b=True, **kw):
        del kw
        self._grid = bool(b)

    def invert_yaxis(self):
        self._invert_y = True

    def set_aspect(self, *a, **kw):
        del a, kw

    def autoscale_view(self, *a, **kw):
        del a, kw

    def add_patch(self, patch):
        self.items.append(patch.item())

    def add_collection(self, coll, **kw):
        del kw
        for it in getattr(coll, "items", []):
            self.items.append(it)

    def get_figure(self):
        return self.fig

    def get_xlim(self):
        return self._xlim or self._data_bounds()[0]

    def get_ylim(self):
        return self._ylim or self._data_bounds()[1]

    # ------------------------------------------------------ bounds
    def _data_bounds(self):
        xs, ys = [], []
        for it in self.items:
            k = it[0]
            if k in ("line", "scatter"):
                xs += it[1]
                ys += it[2]
            elif k == "bar":
                horiz = it[7]
                pos, val, base, w = it[1], it[2], it[3], it[4]
                if horiz:
                    ys += [p - w / 2 for p in pos] + \
                        [p + w / 2 for p in pos]
                    xs += val + base
                else:
                    xs += [p - w / 2 for p in pos] + \
                        [p + w / 2 for p in pos]
                    ys += [v + b for v, b in zip(val, base)] + base
            elif k == "fill":
                xs += it[1]
                ys += it[2] + it[3]
            elif k == "polyfill":
                xs += it[1]
                ys += it[2]
            elif k == "mesh":
                grid = it[1]
                xs += [0.0, float(len(grid[0]) if grid else 1)]
                ys += [0.0, float(len(grid))]
            elif k == "text":
                xs.append(it[1])
                ys.append(it[2])
        if not xs:
            xs = [0.0, 1.0]
        if not ys:
            ys = [0.0, 1.0]
        def pad(lo, hi):
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            m = 0.05 * (hi - lo)
            return (lo - m, hi + m)
        return pad(min(xs), max(xs)), pad(min(ys), max(ys))


def _flt_safe(v):
    try:
        return _flt(v)
    except Exception:
        return []


class _Patch:
    """Rectangle/Circle recorded as scene items."""

    def __init__(self, item):
        self._item = item

    def item(self):
        return self._item


class Rectangle(_Patch):
    def __init__(self, xy, width, height, *, color=None,
                 facecolor=None, alpha=1.0, **kw):
        del kw
        col = _rgb(facecolor if facecolor is not None else color,
                   (127, 127, 127))
        super().__init__(("bar", [xy[0] + width / 2], [height],
                          [xy[1]], width, col, alpha, False))


class Circle(_Patch):
    def __init__(self, xy, radius=1.0, *, color=None, alpha=1.0, **kw):
        del kw
        super().__init__(("scatter", [xy[0]], [xy[1]],
                          [_rgb(color, (127, 127, 127))],
                          [max(radius * 40.0, 9.0)], alpha))


class Figure:
    def __init__(self, figsize=(6.4, 4.8), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.axes = []
        self._suptitle = ""

    def add_axes(self, rect, **kw):
        del kw
        ax = Axes(self, tuple(float(v) for v in rect))
        self.axes.append(ax)
        return ax

    def suptitle(self, s, **kw):
        del kw
        self._suptitle = str(s)

    def tight_layout(self, *a, **kw):
        del a, kw

    def colorbar(self, *a, **kw):
        del a, kw
        return None

    def get(self, *a, **kw):
        del a, kw
        return None

    def savefig(self, path, dpi=None, bbox_inches=None, format=None,
                **kw):
        del bbox_inches, kw
        p = str(path)
        fmt = (format or p.rsplit(".", 1)[-1]).lower()
        if fmt == "svg":
            data = render_svg(self)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(data)
        elif fmt == "png":
            with open(p, "wb") as fh:
                fh.write(render_png(self, dpi or self.dpi))
        else:
            raise ValueError(
                "native plot core writes svg and png; got %r" % fmt)

    def savefig_bytes(self, dpi=None):
        return render_png(self, dpi or self.dpi)


# -------------------------------------------------------- rendering
def _layout(fig, scale=1.0):
    W = int(fig.figsize[0] * fig.dpi * scale)
    H = int(fig.figsize[1] * fig.dpi * scale)
    return W, H


def _ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t0 = math.ceil(lo / step) * step
    out = []
    t = t0
    while t <= hi + 1e-12 * span:
        out.append(round(t, 10))
        t += step
    return out


def _fmt_tick(v):
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return "%.3g" % v


class _Canvas:
    """RGB pixel canvas with line/rect/disc/glyph primitives."""

    def __init__(self, W, H, bg=(255, 255, 255)):
        self.W, self.H = W, H
        self.px = bytearray(W * H * 3)
        r, g, b = bg
        for i in range(0, len(self.px), 3):
            self.px[i] = r
            self.px[i + 1] = g
            self.px[i + 2] = b

    def put(self, x, y, col, alpha=1.0):
        if 0 <= x < self.W and 0 <= y < self.H:
            i = 3 * (y * self.W + x)
            if alpha >= 1.0:
                self.px[i], self.px[i + 1], self.px[i + 2] = col
            else:
                a = alpha
                for k in range(3):
                    self.px[i + k] = int(
                        (1 - a) * self.px[i + k] + a * col[k])

    def rect(self, x0, y0, x1, y1, col, alpha=1.0):
        for y in range(max(0, int(y0)), min(self.H, int(y1) + 1)):
            for x in range(max(0, int(x0)), min(self.W, int(x1) + 1)):
                self.put(x, y, col, alpha)

    def line(self, x0, y0, x1, y1, col, width=1.0, alpha=1.0,
             dash=None):
        # Bresenham with square pen; dash = (on, off) in px
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        w = max(1, int(round(width)))
        half = w // 2
        n = 0
        on = True
        while True:
            if dash:
                period = dash[0] + dash[1]
                on = (n % period) < dash[0]
            if on:
                for oy in range(-half, half + 1):
                    for ox in range(-half, half + 1):
                        self.put(x0 + ox, y0 + oy, col, alpha)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            n += 1

    def disc(self, cx, cy, r, col, alpha=1.0):
        rr = int(math.ceil(r))
        for y in range(-rr, rr + 1):
            for x in range(-rr, rr + 1):
                if x * x + y * y <= r * r:
                    self.put(int(cx) + x, int(cy) + y, col, alpha)

    def text(self, x, y, s, col, size=10.0, ha="left"):
        sc = max(1, int(round(size / 7.0)))
        wpx = len(s) * 6 * sc
        if ha == "center":
            x -= wpx // 2
        elif ha == "right":
            x -= wpx
        cx = int(x)
        for ch in s:
            glyph = _FONT.get(ch, _FONT["?"])
            for colidx, bits in enumerate(glyph):
                for row in range(7):
                    if bits & (1 << row):
                        for oy in range(sc):
                            for ox in range(sc):
                                self.put(cx + colidx * sc + ox,
                                         int(y) + row * sc + oy, col)
            cx += 6 * sc


def render_png(fig, dpi=None):
    scale = (dpi / fig.dpi) if dpi else 1.0
    W, H = _layout(fig, scale)
    cv = _Canvas(W, H)
    if fig._suptitle:
        cv.text(W // 2, 4, fig._suptitle, (0, 0, 0), 12 * scale,
                ha="center")
    for ax in (fig.axes or [Axes(fig, (0.1, 0.1, 0.85, 0.8))]):
        _render_axes_png(cv, ax, W, H, scale)
    return _encode_png(cv)


def _render_axes_png(cv, ax, W, H, scale):
    fx, fy, fw, fh = ax.rect
    x0 = int(fx * W)
    y1 = int((1 - fy) * H)                     # bottom
    x1 = int((fx + fw) * W)
    y0 = int((1 - fy - fh) * H)                # top
    (dx0, dx1), (dy0, dy1) = (ax._xlim or ax._data_bounds()[0]), \
        (ax._ylim or ax._data_bounds()[1])
    if ax._invert_y:
        dy0, dy1 = dy1, dy0

    def X(v):
        return x0 + (v - dx0) / (dx1 - dx0 or 1.0) * (x1 - x0)

    def Y(v):
        return y1 - (v - dy0) / (dy1 - dy0 or 1.0) * (y1 - y0)

    black = (0, 0, 0)
    lgray = (200, 200, 200)
    xticks = ax._xticks if ax._xticks is not None else _ticks(dx0, dx1)
    yticks = ax._yticks if ax._yticks is not None else _ticks(dy0, dy1)
    if ax._grid:
        for t in xticks:
            cv.line(X(t), y0, X(t), y1, lgray, 1, 0.6)
        for t in yticks:
            cv.line(x0, Y(t), x1, Y(t), lgray, 1, 0.6)

    for it in ax.items:
        k = it[0]
        if k == "line":
            _, xs, ys, col, wdt, style, alpha = it
            dash = {"--": (6, 4), ":": (2, 3),
                    "-.": (6, 3)}.get(style)
            for i in range(len(xs) - 1):
                cv.line(X(xs[i]), Y(ys[i]), X(xs[i + 1]),
                        Y(ys[i + 1]), col, wdt * scale, alpha, dash)
        elif k == "scatter":
            _, xs, ys, cols, sizes, alpha = it
            for xv, yv, c, sv in zip(xs, ys, cols, sizes):
                cv.disc(X(xv), Y(yv), max(1.5, math.sqrt(sv) / 2)
                        * scale, c, alpha)
        elif k == "bar":
            _, pos, val, base, w, col, alpha, horiz = it
            for p, v, b in zip(pos, val, base):
                if horiz:
                    cv.rect(min(X(b), X(b + v)), Y(p + w / 2),
                            max(X(b), X(b + v)), Y(p - w / 2),
                            col, alpha)
                else:
                    cv.rect(X(p - w / 2), min(Y(b), Y(b + v)),
                            X(p + w / 2), max(Y(b), Y(b + v)),
                            col, alpha)
        elif k == "fill":
            _, xs, a, b, col, alpha = it
            for i in range(len(xs) - 1):
                for xx in range(int(X(xs[i])), int(X(xs[i + 1])) + 1):
                    f = 0 if X(xs[i + 1]) == X(xs[i]) else \
                        (xx - X(xs[i])) / (X(xs[i + 1]) - X(xs[i]))
                    ya = a[i] + f * (a[i + 1] - a[i])
                    yb = b[i] + f * (b[i + 1] - b[i])
                    cv.line(xx, Y(ya), xx, Y(yb), col, 1, alpha)
        elif k == "hline":
            _, yv, col, wdt, style, alpha = it
            dash = {"--": (6, 4), ":": (2, 3)}.get(style)
            cv.line(x0, Y(yv), x1, Y(yv), col, wdt * scale, alpha,
                    dash)
        elif k == "vline":
            _, xv, col, wdt, style, alpha = it
            dash = {"--": (6, 4), ":": (2, 3)}.get(style)
            cv.line(X(xv), y0, X(xv), y1, col, wdt * scale, alpha,
                    dash)
        elif k == "mesh":
            _, grid, extent, alpha = it
            nr = len(grid)
            nc = len(grid[0]) if nr else 0
            for r in range(nr):
                for c in range(nc):
                    cv.rect(x0 + c * (x1 - x0) / nc,
                            y0 + r * (y1 - y0) / nr,
                            x0 + (c + 1) * (x1 - x0) / nc,
                            y0 + (r + 1) * (y1 - y0) / nr,
                            grid[r][c], alpha)
        elif k == "polyfill":
            _, xs, ys, col, alpha = it
            n = len(xs)
            ymin, ymax = int(min(Y(v) for v in ys)), \
                int(max(Y(v) for v in ys))
            pxy = [(X(a), Y(b)) for a, b in zip(xs, ys)]
            for yy in range(max(y0, ymin), min(y1, ymax) + 1):
                cuts = []
                for i in range(n):
                    xa, ya = pxy[i]
                    xb, yb = pxy[(i + 1) % n]
                    if (ya <= yy < yb) or (yb <= yy < ya):
                        cuts.append(xa + (yy - ya) * (xb - xa)
                                    / ((yb - ya) or 1.0))
                cuts.sort()
                for j in range(0, len(cuts) - 1, 2):
                    cv.line(cuts[j], yy, cuts[j + 1], yy, col, 1,
                            alpha)
        elif k == "text":
            _, xv, yv, s, size, col, ha, va = it
            del va
            cv.text(X(xv), Y(yv), s, col, size * scale, ha=ha)

    # frame, ticks, labels
    cv.line(x0, y0, x1, y0, black)
    cv.line(x0, y1, x1, y1, black)
    cv.line(x0, y0, x0, y1, black)
    cv.line(x1, y0, x1, y1, black)
    xlabels = ax._xticklabels if ax._xticklabels is not None else \
        [_fmt_tick(t) for t in xticks]
    for t, lab in zip(xticks, xlabels):
        if dx0 <= t <= dx1 or ax._xticks is not None:
            cv.line(X(t), y1, X(t), y1 + 4, black)
            cv.text(X(t), y1 + 6, lab, black, 8 * scale, ha="center")
    ylabels = ax._yticklabels if ax._yticklabels is not None else \
        [_fmt_tick(t) for t in yticks]
    for t, lab in zip(yticks, ylabels):
        if dy0 <= t <= dy1 or ax._yticks is not None:
            cv.line(x0 - 4, Y(t), x0, Y(t), black)
            cv.text(x0 - 6, Y(t) - 3, lab, black, 8 * scale,
                    ha="right")
    if ax.title:
        cv.text((x0 + x1) // 2, max(0, y0 - 14), ax.title, black,
                11 * scale, ha="center")
    if ax.xlabel:
        cv.text((x0 + x1) // 2, min(cv.H - 8, y1 + 18), ax.xlabel,
                black, 9 * scale, ha="center")
    if ax.ylabel:
        cv.text(max(2, x0 - 40), max(0, y0 - 14), ax.ylabel, black,
                9 * scale)
    if ax._show_legend and ax._legend:
        ly = y0 + 6
        for lab, col in ax._legend[:8]:
            cv.rect(x1 - 90, ly, x1 - 78, ly + 8, col)
            cv.text(x1 - 74, ly, lab[:14], black, 8 * scale)
            ly += 14


def _encode_png(cv):
    """Minimal PNG encoder: 8-bit RGB, filter 0 (RFC 2083 sec. 6)."""
    raw = bytearray()
    stride = cv.W * 3
    for y in range(cv.H):
        raw.append(0)
        raw += cv.px[y * stride:(y + 1) * stride]

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data)
                               & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", cv.W, cv.H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def render_svg(fig):
    png_like = []  # simple approach: draw same primitives as SVG
    W, H = _layout(fig)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
           'height="%d" viewBox="0 0 %d %d">' % (W, H, W, H),
           '<rect width="100%" height="100%" fill="white"/>']
    if fig._suptitle:
        out.append('<text x="%d" y="14" text-anchor="middle" '
                   'font-size="13">%s</text>'
                   % (W // 2, _esc(fig._suptitle)))
    for ax in fig.axes:
        out.append(_render_axes_svg(ax, W, H))
    out.append("</svg>")
    del png_like
    return "\n".join(out)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _render_axes_svg(ax, W, H):
    # reuse the raster path by rendering data into SVG primitives
    fx, fy, fw, fh = ax.rect
    x0, x1 = fx * W, (fx + fw) * W
    y1, y0 = (1 - fy) * H, (1 - fy - fh) * H
    (dx0, dx1), (dy0, dy1) = (ax._xlim or ax._data_bounds()[0]), \
        (ax._ylim or ax._data_bounds()[1])
    if ax._invert_y:
        dy0, dy1 = dy1, dy0

    def X(v):
        return x0 + (v - dx0) / (dx1 - dx0 or 1.0) * (x1 - x0)

    def Y(v):
        return y1 - (v - dy0) / (dy1 - dy0 or 1.0) * (y1 - y0)

    def rgb(c):
        return "rgb(%d,%d,%d)" % c

    o = []
    for it in ax.items:
        k = it[0]
        if k == "line":
            _, xs, ys, col, wdt, style, alpha = it
            pts = " ".join("%.2f,%.2f" % (X(a), Y(b))
                           for a, b in zip(xs, ys))
            dash = {"--": "6,4", ":": "2,3", "-.": "6,3"}.get(style, "")
            o.append('<polyline fill="none" stroke="%s" '
                     'stroke-width="%.2f" stroke-opacity="%.2f"%s '
                     'points="%s"/>'
                     % (rgb(col), wdt, alpha,
                        ' stroke-dasharray="%s"' % dash if dash else "",
                        pts))
        elif k == "scatter":
            _, xs, ys, cols, sizes, alpha = it
            for a, b, c, sv in zip(xs, ys, cols, sizes):
                o.append('<circle cx="%.2f" cy="%.2f" r="%.2f" '
                         'fill="%s" fill-opacity="%.2f"/>'
                         % (X(a), Y(b), max(1.5, math.sqrt(sv) / 2),
                            rgb(c), alpha))
        elif k == "bar":
            _, pos, val, base, w, col, alpha, horiz = it
            for p, v, b in zip(pos, val, base):
                if horiz:
                    xa, xb = sorted((X(b), X(b + v)))
                    ya, yb = sorted((Y(p + w / 2), Y(p - w / 2)))
                else:
                    xa, xb = sorted((X(p - w / 2), X(p + w / 2)))
                    ya, yb = sorted((Y(b), Y(b + v)))
                o.append('<rect x="%.2f" y="%.2f" width="%.2f" '
                         'height="%.2f" fill="%s" '
                         'fill-opacity="%.2f"/>'
                         % (xa, ya, xb - xa, yb - ya, rgb(col), alpha))
        elif k == "fill":
            _, xs, a, b, col, alpha = it
            fwd = ["%.2f,%.2f" % (X(p), Y(q)) for p, q in zip(xs, a)]
            back = ["%.2f,%.2f" % (X(p), Y(q))
                    for p, q in zip(xs[::-1], b[::-1])]
            o.append('<polygon fill="%s" fill-opacity="%.2f" '
                     'points="%s"/>' % (rgb(col), alpha,
                                        " ".join(fwd + back)))
        elif k == "hline":
            _, yv, col, wdt, style, alpha = it
            o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                     'stroke="%s" stroke-width="%.1f" '
                     'stroke-dasharray="6,4" stroke-opacity="%.2f"/>'
                     % (x0, Y(yv), x1, Y(yv), rgb(col), wdt, alpha))
        elif k == "vline":
            _, xv, col, wdt, style, alpha = it
            o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                     'stroke="%s" stroke-width="%.1f" '
                     'stroke-dasharray="6,4" stroke-opacity="%.2f"/>'
                     % (X(xv), y0, X(xv), y1, rgb(col), wdt, alpha))
        elif k == "mesh":
            _, grid, extent, alpha = it
            nr = len(grid)
            nc = len(grid[0]) if nr else 0
            cw = (x1 - x0) / (nc or 1)
            ch = (y1 - y0) / (nr or 1)
            for r in range(nr):
                for c in range(nc):
                    o.append('<rect x="%.2f" y="%.2f" width="%.2f" '
                             'height="%.2f" fill="%s" '
                             'fill-opacity="%.2f"/>'
                             % (x0 + c * cw, y0 + r * ch, cw + 0.5,
                                ch + 0.5, rgb(grid[r][c]), alpha))
        elif k == "polyfill":
            _, xs, ys, col, alpha = it
            pts = " ".join("%.2f,%.2f" % (X(a), Y(b))
                           for a, b in zip(xs, ys))
            o.append('<polygon fill="%s" fill-opacity="%.2f" '
                     'points="%s"/>' % (rgb(col), alpha, pts))
        elif k == "text":
            _, xv, yv, s, size, col, ha, va = it
            anchor = {"left": "start", "center": "middle",
                      "right": "end"}.get(ha, "start")
            o.append('<text x="%.1f" y="%.1f" font-size="%.1f" '
                     'fill="%s" text-anchor="%s">%s</text>'
                     % (X(xv), Y(yv), size, rgb(col), anchor,
                        _esc(s)))

    xticks = ax._xticks if ax._xticks is not None else _ticks(dx0, dx1)
    yticks = ax._yticks if ax._yticks is not None else _ticks(dy0, dy1)
    xlabels = ax._xticklabels if ax._xticklabels is not None else \
        [_fmt_tick(t) for t in xticks]
    ylabels = ax._yticklabels if ax._yticklabels is not None else \
        [_fmt_tick(t) for t in yticks]
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
             'fill="none" stroke="black"/>'
             % (x0, y0, x1 - x0, y1 - y0))
    for t, lab in zip(xticks, xlabels):
        o.append('<text x="%.1f" y="%.1f" font-size="9" '
                 'text-anchor="middle">%s</text>'
                 % (X(t), y1 + 12, _esc(lab)))
    for t, lab in zip(yticks, ylabels):
        o.append('<text x="%.1f" y="%.1f" font-size="9" '
                 'text-anchor="end">%s</text>'
                 % (x0 - 4, Y(t) + 3, _esc(lab)))
    if ax.title:
        o.append('<text x="%.1f" y="%.1f" font-size="12" '
                 'text-anchor="middle">%s</text>'
                 % ((x0 + x1) / 2, y0 - 6, _esc(ax.title)))
    if ax.xlabel:
        o.append('<text x="%.1f" y="%.1f" font-size="10" '
                 'text-anchor="middle">%s</text>'
                 % ((x0 + x1) / 2, y1 + 26, _esc(ax.xlabel)))
    if ax.ylabel:
        o.append('<text x="%.1f" y="%.1f" font-size="10" '
                 'transform="rotate(-90 %.1f %.1f)" '
                 'text-anchor="middle">%s</text>'
                 % (x0 - 30, (y0 + y1) / 2, x0 - 30, (y0 + y1) / 2,
                    _esc(ax.ylabel)))
    if ax._show_legend and ax._legend:
        ly = y0 + 12
        for lab, col in ax._legend[:8]:
            o.append('<rect x="%.1f" y="%.1f" width="10" height="8" '
                     'fill="%s"/>' % (x1 - 90, ly - 8, rgb(col)))
            o.append('<text x="%.1f" y="%.1f" font-size="9">%s</text>'
                     % (x1 - 76, ly, _esc(lab[:16])))
            ly += 13
    return "\n".join(o)


# ------------------------------------------------------ pyplot API
_open_figs = []


def figure(figsize=(6.4, 4.8), dpi=100, **kw):
    del kw
    f = Figure(figsize, dpi)
    _open_figs.append(f)
    return f


def subplots(nrows=1, ncols=1, figsize=None, sharex=False,
             sharey=False, dpi=100, squeeze=True, **kw):
    del sharex, sharey, kw
    if figsize is None:
        figsize = (6.4 * ncols, 4.8 * nrows)
    fig = figure(figsize, dpi)
    grid = []
    mL, mR, mB, mT = 0.10, 0.04, 0.11, 0.08
    cw = (1.0 - mL - mR) / ncols
    ch = (1.0 - mT - mB) / nrows
    for r in range(nrows):
        row = []
        for c in range(ncols):
            rect = (mL + c * cw + 0.02, mB + (nrows - 1 - r) * ch
                    + 0.02, cw - 0.05, ch - 0.09)
            row.append(fig.add_axes(rect))
        grid.append(row)
    if squeeze:
        if nrows == 1 and ncols == 1:
            return fig, grid[0][0]
        if nrows == 1:
            return fig, grid[0]
        if ncols == 1:
            return fig, [r[0] for r in grid]
    return fig, grid


def close(fig=None):
    if fig in ("all", None):
        _open_figs.clear()
    elif fig in _open_figs:
        _open_figs.remove(fig)


def gca():
    if not _open_figs:
        figure()
    f = _open_figs[-1]
    if not f.axes:
        f.add_axes((0.1, 0.11, 0.86, 0.81))
    return f.axes[-1]


def gcf():
    if not _open_figs:
        figure()
    return _open_figs[-1]


def tight_layout(*a, **kw):
    del a, kw


def colorbar(*a, **kw):
    del a, kw


def savefig(path, **kw):
    gcf().savefig(path, **kw)


def show(*a, **kw):
    del a, kw


def suptitle(s, **kw):
    gcf().suptitle(s, **kw)


def title(s, **kw):
    gca().set_title(s, **kw)


def xlabel(s, **kw):
    gca().set_xlabel(s, **kw)


def ylabel(s, **kw):
    gca().set_ylabel(s, **kw)


def plot(*a, **kw):
    return gca().plot(*a, **kw)


def scatter(*a, **kw):
    return gca().scatter(*a, **kw)


def bar(*a, **kw):
    return gca().bar(*a, **kw)


def hist(*a, **kw):
    return gca().hist(*a, **kw)


def legend(*a, **kw):
    gca().legend(*a, **kw)


def grid(*a, **kw):
    gca().grid(*a, **kw)


def xlim(*a, **kw):
    gca().set_xlim(*a, **kw)


def ylim(*a, **kw):
    gca().set_ylim(*a, **kw)


# matplotlib.patches surface used by morie.viz
class _PatchesNS:
    Rectangle = Rectangle
    Circle = Circle


patches = _PatchesNS()
pyplot = None  # set below for "import _plot_core as plt" symmetry


class PolyCollection:
    """List of filled polygons (matplotlib.collections surface).

    Each polygon becomes a scene fill; Axes.add_collection copies the
    items in.
    """

    def __init__(self, verts, *, facecolors=None, edgecolors=None,
                 alpha=0.5, **kw):
        del edgecolors, kw
        self.items = []
        for i, poly in enumerate(verts):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            if isinstance(facecolors, (list, tuple)) and facecolors \
                    and not isinstance(facecolors, str) \
                    and not isinstance(facecolors[0], (int, float)):
                col = _rgb(facecolors[i % len(facecolors)])
            else:
                col = _rgb(facecolors, (31, 119, 180))
            self.items.append(("polyfill", xs, ys, col, alpha))

    def set_alpha(self, a):
        self.items = [it[:4] + (float(a),) for it in self.items]


class Normalize:
    """(v - vmin) / (vmax - vmin), clipped to [0, 1]."""

    def __init__(self, vmin=0.0, vmax=1.0, clip=False):
        self.vmin, self.vmax = float(vmin), float(vmax)
        del clip

    def __call__(self, v):
        span = (self.vmax - self.vmin) or 1.0
        return min(max((float(v) - self.vmin) / span, 0.0), 1.0)


class _PathEffectsNS:
    @staticmethod
    def withStroke(*a, **kw):
        del a, kw
        return None

    @staticmethod
    def Stroke(*a, **kw):
        del a, kw
        return None

    @staticmethod
    def Normal(*a, **kw):
        del a, kw
        return None


patheffects = _PathEffectsNS()


def use(backend, *a, **kw):
    """matplotlib.use shim: the native core has exactly one backend."""
    del backend, a, kw
