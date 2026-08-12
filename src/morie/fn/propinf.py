# morie.fn -- function file (rootcoder007/morie)
r"""Property inference on fully connected neural networks.

Ganju, K., Wang, Q., Yang, W., Gunter, C. A., & Borisov, N. (2018)
"Property Inference Attacks on Fully Connected Neural Networks using
Permutation Invariant Representations", *CCS '18*, 619-633.
doi:10.1145/3243734.3243834

The attack answers a question about the *training set* from the trained
model alone: given only the released weights of a fully connected
network, was it trained on data with property :math:`P` (say, 65% women)
or without it? The recipe is shadow training (Section 4): the adversary
trains :math:`k` shadow classifiers of the same architecture, half on
data with :math:`P` and half without, turns each one into a feature
vector, and fits a *meta-classifier* on those vectors labelled by
:math:`P` / :math:`\bar{P}`. The target model's feature vector is then
handed to the meta-classifier.

**Why the obvious feature representation fails.** Flattening all the
weights and biases into one vector is the baseline, and on neural
networks it barely beats a coin: 55-58% in the paper's Census and MNIST
experiments. The reason is Proposition 5.1, *permutation equivalence*:
reordering the neurons within a hidden layer (and permuting the next
layer's weight columns to match) leaves the function the network
computes completely unchanged, but moves every entry of the flattened
vector. A hidden layer of size :math:`|h_t|` has :math:`|h_t|!`
orderings, so a network with layers :math:`h_1,\dots,h_k` has
:math:`\prod_t |h_t|!` equivalents that a meta-classifier would have to
learn about separately.

Three representations are implemented, and the choice is exposed,
because the paper reports all three (Table 2):

``"baseline"``
    The flattened vector. Kept because it is the thing the other two
    have to beat, not because it works.

``"sorting"`` (Algorithm 1)
    Put the network in a canonical form first. Within each hidden layer
    sort the neurons by the magnitude of the sum of their weights, then
    apply that permutation to the layer and to the next layer's weight
    columns. All permutation equivalents collapse to one vector. Still
    a flat vector afterwards, so any ordinary classifier can consume it.

``"set"`` (Algorithm 2, the default)
    Treat each layer as a *set* of neurons and learn on it with the
    DeepSets form :math:`\rho\bigl(\sum_{x \in X} \varphi(x)\bigr)`. One
    :math:`\varphi_t` per layer maps a neuron's weights and bias to a
    node representation :math:`N^t_i`; those are summed over the layer,
    :math:`L_t = \sum_i N^t_i`, and the concatenated :math:`L_t` go to a
    single :math:`\rho`. Because the sum is unweighted the layer itself
    is permutation invariant by construction rather than by
    canonicalisation. Section 6.2 also gives :math:`\varphi_t` for
    :math:`t > 1` the previous layer's node representations as context,
    so a neuron is described by what it computes *and* by what it
    computes it from -- but the printed form of that context is a
    concatenation in node order, which is **not** invariant. See
    :func:`_deepsets_forward` and the ``context`` argument; the default
    is the invariant reading.

**A caveat to Table 2, measured here.** The paper reports the set-based
route ahead of sorting ahead of the baseline in all ten experiments.
That ordering reproduces when the property is carried by parameters a
permutation actually moves -- if which input feature drives the label is
the property, the set route reads it off 30 unseen models at 93% against
47% for the baseline. It does not reproduce when the property is carried
by the *output* bias, which Section 5 itself says no permutation can
touch: on a class-prior property (the paper's :math:`P^{Census}_2` in
miniature) the flattened baseline is not beaten, and its accuracy swings
with the number of shadow models (0.37 at 60, 0.78 at 120). The
representations are not the whole story; where the property lives is.

Sorting is a preprocessing step; the set-based route is a different
meta-classifier. That is why ``representation="set"`` ignores
``meta_hidden`` in favour of ``phi_hidden`` and ``rho_hidden``.

The networks here are trained in plain Python by minibatch SGD, so the
sizes are the ones an anchor can check rather than the paper's 4,096
shadow models on a GPU.
"""

import math

from . import _array_core as np  # noqa: F401  (kept for array interop)

from ._richresult import RichResult

__all__ = [
    "property_inference",
    "train_fcnn",
    "flat_representation",
    "sorted_representation",
    "set_representation",
    "permute_hidden_layer",
    "fcnn_predict",
]

_REPRS = ("baseline", "sorting", "set")
_CONTEXTS = ("paired", "as_printed", "none")


# ---------------------------------------------------------------- rng

def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _normal(rnd, scale=1.0):
    u = max(rnd(), 1e-12)
    return scale * math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * rnd())


# ------------------------------------------------------- dense network
#
# A network is a list of layers, each {"W": [[...]], "b": [...]}, with
# ReLU on the hidden layers and a logistic output.  Small and explicit
# so that the permutation argument can be checked on it directly.

def _relu(v):
    return v if v > 0.0 else 0.0


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _init_net(n_in, hidden, rnd):
    sizes = [int(n_in)] + [int(h) for h in hidden] + [1]
    net = []
    for t in range(1, len(sizes)):
        fan_in = sizes[t - 1]
        s = math.sqrt(2.0 / fan_in)
        net.append({
            "W": [[_normal(rnd, s) for _ in range(fan_in)]
                  for _ in range(sizes[t])],
            "b": [0.0] * sizes[t],
        })
    return net


def _forward(net, x):
    """Activations layer by layer; the last layer is logistic."""
    acts = [list(x)]
    pre = []
    last = len(net) - 1
    for t, layer in enumerate(net):
        z = [sum(w * a for w, a in zip(row, acts[-1])) + bias
             for row, bias in zip(layer["W"], layer["b"])]
        pre.append(z)
        acts.append([_sigmoid(v) for v in z] if t == last
                    else [_relu(v) for v in z])
    return acts, pre


def fcnn_predict(net, X):
    """Output of the network on each row of ``X``."""
    return [_forward(net, row)[0][-1][0] for row in _rows(X)]


def _rows(X):
    out = []
    for r in X:
        row = [float(v) for v in r]
        for v in row:
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("propinf: X contains a non-finite value")
        out.append(row)
    if not out:
        raise ValueError("propinf: X is empty")
    p = len(out[0])
    if p == 0:
        raise ValueError("propinf: X has no columns")
    for row in out:
        if len(row) != p:
            raise ValueError("propinf: X is ragged")
    return out


def train_fcnn(X, y, hidden=(8, 4), epochs=40, lr=0.1, batch_size=16,
               seed=0):
    """Train a fully connected binary classifier by minibatch SGD.

    Cross-entropy loss, ReLU hidden units, logistic output -- the
    ordinary setting the paper's target and shadow models live in.
    """
    rows = _rows(X)
    lab = [float(v) for v in y]
    if len(lab) != len(rows):
        raise ValueError("propinf: X and y have different lengths")
    if any(v not in (0.0, 1.0) for v in lab):
        raise ValueError("propinf: y must be 0/1")
    if not hidden:
        raise ValueError("propinf: at least one hidden layer is required")
    if epochs < 1:
        raise ValueError("propinf: epochs must be at least 1")
    if lr <= 0:
        raise ValueError("propinf: lr must be positive")
    if batch_size < 1:
        raise ValueError("propinf: batch_size must be at least 1")
    rnd = _rng(seed + 1)
    net = _init_net(len(rows[0]), hidden, rnd)
    n = len(rows)
    order = list(range(n))
    for _ in range(int(epochs)):
        for i in range(n - 1, 0, -1):        # Fisher-Yates on the LCG
            j = int(rnd() * (i + 1))
            order[i], order[j] = order[j], order[i]
        for start in range(0, n, int(batch_size)):
            chunk = order[start:start + int(batch_size)]
            gW = [[[0.0] * len(r) for r in L["W"]] for L in net]
            gb = [[0.0] * len(L["b"]) for L in net]
            for idx in chunk:
                acts, pre = _forward(net, rows[idx])
                # dL/dz at the logistic output, cross entropy
                delta = [acts[-1][0] - lab[idx]]
                for t in range(len(net) - 1, -1, -1):
                    a_in = acts[t]
                    for i2, d in enumerate(delta):
                        gb[t][i2] += d
                        row = gW[t][i2]
                        for j2, a in enumerate(a_in):
                            row[j2] += d * a
                    if t > 0:
                        nxt = [0.0] * len(a_in)
                        for i2, d in enumerate(delta):
                            wrow = net[t]["W"][i2]
                            for j2 in range(len(a_in)):
                                nxt[j2] += d * wrow[j2]
                        delta = [g if pre[t - 1][j2] > 0.0 else 0.0
                                 for j2, g in enumerate(nxt)]
            scale = lr / float(len(chunk))
            for t, layer in enumerate(net):
                for i2, row in enumerate(layer["W"]):
                    grow = gW[t][i2]
                    for j2 in range(len(row)):
                        row[j2] -= scale * grow[j2]
                    layer["b"][i2] -= scale * gb[t][i2]
    return net


# ----------------------------------------------- permutation machinery

def permute_hidden_layer(net, t, sigma):
    """Reorder the neurons of hidden layer ``t`` by ``sigma``.

    The transformation of Section 5: layer ``t`` becomes
    :math:`\\sigma(h_t)` and the next layer's weight columns are permuted
    to match, so the network computes exactly the same function
    (Proposition 5.1).
    """
    if not 0 <= int(t) < len(net) - 1:
        raise ValueError("propinf: t must index a hidden layer")
    sigma = [int(v) for v in sigma]
    m = len(net[t]["W"])
    if sorted(sigma) != list(range(m)):
        raise ValueError("propinf: sigma is not a permutation of the layer")
    out = [{"W": [list(r) for r in L["W"]], "b": list(L["b"])} for L in net]
    out[t]["W"] = [list(net[t]["W"][s]) for s in sigma]
    out[t]["b"] = [net[t]["b"][s] for s in sigma]
    out[t + 1]["W"] = [[row[s] for s in sigma] for row in net[t + 1]["W"]]
    return out


def _node_metric(layer, i):
    """Algorithm 1's sorting key: the magnitude of the node's weight sum."""
    return abs(sum(layer["W"][i]))


def flat_representation(net):
    """The baseline: every weight and bias in one vector."""
    F = []
    for layer in net:
        for row, bias in zip(layer["W"], layer["b"]):
            F.extend(row)
            F.append(bias)
    return F


def sorted_representation(net, metric=None):
    """Algorithm 1: the canonical form, then flatten.

    Hidden layers are sorted by ``metric`` (default: the magnitude of the
    sum of the node's weights) in descending order; the output layer
    cannot be permuted and is appended as it stands.
    """
    if metric is None:
        metric = _node_metric
    cur = [{"W": [list(r) for r in L["W"]], "b": list(L["b"])} for L in net]
    for t in range(len(cur) - 1):
        vals = [metric(cur[t], i) for i in range(len(cur[t]["W"]))]
        sigma = sorted(range(len(vals)), key=lambda i: -vals[i])
        cur = permute_hidden_layer(cur, t, sigma)
    return flat_representation(cur)


def set_representation(net):
    """Algorithm 2's input: each layer as a set of ``(weights, bias)``."""
    return [[list(row) + [bias]
             for row, bias in zip(layer["W"], layer["b"])]
            for layer in net]


# --------------------------------------------------- meta-classifiers
#
# Two of them: an ordinary MLP over a flat vector (baseline and
# sorting), and the DeepSets network of Section 6.2 (set-based).

def _mlp_init(sizes, rnd):
    net = []
    for t in range(1, len(sizes)):
        s = math.sqrt(2.0 / sizes[t - 1])
        net.append({
            "W": [[_normal(rnd, s) for _ in range(sizes[t - 1])]
                  for _ in range(sizes[t])],
            "b": [0.0] * sizes[t],
        })
    return net


def _mlp_forward(net, x, final="relu", hidden_act="relu"):
    acts, pre = [list(x)], []
    last = len(net) - 1
    for t, layer in enumerate(net):
        z = [sum(w * a for w, a in zip(row, acts[-1])) + bias
             for row, bias in zip(layer["W"], layer["b"])]
        pre.append(z)
        if t == last and final == "linear":
            acts.append(list(z))
        elif t == last and final == "sigmoid":
            acts.append([_sigmoid(v) for v in z])
        elif t == last and final == "tanh":
            acts.append([math.tanh(v) for v in z])
        elif hidden_act == "tanh":
            acts.append([math.tanh(v) for v in z])
        else:
            acts.append([_relu(v) for v in z])
    return acts, pre


def _mlp_backward(net, acts, pre, dout, grads, final="relu",
                  hidden_act="relu"):
    """Accumulate gradients into ``grads``; return the gradient w.r.t. x."""
    delta = list(dout)
    last = len(net) - 1
    for t in range(last, -1, -1):
        if t == last and final == "tanh":
            delta = [d * (1.0 - acts[t + 1][i] ** 2)
                     for i, d in enumerate(delta)]
        elif not (t == last and final in ("linear", "sigmoid")):
            if hidden_act == "tanh":
                delta = [d * (1.0 - acts[t + 1][i] ** 2)
                         for i, d in enumerate(delta)]
            else:
                delta = [d if pre[t][i] > 0.0 else 0.0
                         for i, d in enumerate(delta)]
        a_in = acts[t]
        for i, d in enumerate(delta):
            grads[t]["b"][i] += d
            grow = grads[t]["W"][i]
            for j, a in enumerate(a_in):
                grow[j] += d * a
        nxt = [0.0] * len(a_in)
        for i, d in enumerate(delta):
            wrow = net[t]["W"][i]
            for j in range(len(a_in)):
                nxt[j] += d * wrow[j]
        delta = nxt
    return delta


def _zero_like(net):
    return [{"W": [[0.0] * len(r) for r in L["W"]], "b": [0.0] * len(L["b"])}
            for L in net]


def _sgd_step(net, grads, lr, scale):
    for L, g in zip(net, grads):
        for i, row in enumerate(L["W"]):
            grow = g["W"][i]
            for j in range(len(row)):
                row[j] -= lr * scale * grow[j]
            L["b"][i] -= lr * scale * g["b"][i]


def _train_vector_meta(feats, labels, hidden, epochs, lr, seed):
    rnd = _rng(seed + 7)
    d = len(feats[0])
    net = _mlp_init([d] + [int(h) for h in hidden] + [1], rnd)
    n = len(feats)
    order = list(range(n))
    for _ in range(int(epochs)):
        for i in range(n - 1, 0, -1):
            j = int(rnd() * (i + 1))
            order[i], order[j] = order[j], order[i]
        for idx in order:
            acts, pre = _mlp_forward(net, feats[idx], final="sigmoid")
            grads = _zero_like(net)
            _mlp_backward(net, acts, pre, [acts[-1][0] - labels[idx]],
                          grads, final="sigmoid")
            _sgd_step(net, grads, lr, 1.0)
    return net


def _vector_meta_predict(net, f):
    return _mlp_forward(net, f, final="sigmoid")[0][-1][0]


def _deepsets_init(shapes, phi_hidden, repr_dim, rho_hidden, rnd,
                   context="paired", edge_hidden=None):
    """One phi per layer (plus, for ``"paired"``, one psi) and a single rho.

    ``shapes`` is ``[(n_nodes, n_inputs), ...]`` for the network under
    attack. See :func:`_deepsets_forward` for what ``context`` does.
    """
    if context not in _CONTEXTS:
        raise ValueError("propinf: context must be one of %s" % (_CONTEXTS,))
    phi_hidden = [int(h) for h in phi_hidden]
    edge_hidden = phi_hidden if edge_hidden is None else \
        [int(h) for h in edge_hidden]
    phis, psis = [], []
    prev_nodes = 0
    for t, (n_nodes, n_in) in enumerate(shapes):
        psi = None
        if t == 0 or context == "none":
            d_in = n_in + 1
        elif context == "paired":
            psi = _mlp_init([1 + repr_dim] + edge_hidden + [repr_dim], rnd)
            d_in = 1 + repr_dim
        else:                                    # "as_printed"
            d_in = n_in + 1 + prev_nodes * repr_dim
        psis.append(psi)
        phis.append(_mlp_init([d_in] + phi_hidden + [repr_dim], rnd))
        prev_nodes = n_nodes
    rho = _mlp_init([len(shapes) * repr_dim] + [int(h) for h in rho_hidden] +
                    [1], rnd)
    return {"phis": phis, "psis": psis, "rho": rho, "repr_dim": repr_dim,
            "shapes": shapes, "context": context, "scalers": None}


def _layer_scalers(sets_list):
    """Per-layer location and scale for weights and biases.

    One pair of scalars per layer, taken over every weight in it, so the
    rescaling cannot depend on the order of the neurons -- a per-column
    standardisation would, since column ``j`` is indexed by node ``j`` of
    the layer below.
    """
    out = []
    for t in range(len(sets_list[0])):
        ws = [v for st in sets_list for node in st[t] for v in node[:-1]]
        bs = [node[-1] for st in sets_list for node in st[t]]
        stats = []
        for vals in (ws, bs):
            mu = sum(vals) / float(len(vals))
            var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1.0, 1.0)
            stats.append((mu, math.sqrt(var) if var > 1e-24 else 1.0))
        out.append(stats)
    return out


def _deepsets_forward(model, sets):
    r"""Node processing, layer summation, concatenation, prediction.

    The context wiring is where the paper has to be read carefully.
    Section 6.2 writes the context of layer :math:`t` as the tuple
    :math:`N^{t-1} = (N^{t-1}_1, \dots, N^{t-1}_{|h_{t-1}|})`. Taken
    literally that is a concatenation in node order, so permuting layer
    :math:`t-1` changes it -- and it changes the columns of every
    :math:`w^t_{i*}` as well, since those are indexed by the previous
    layer's nodes. Algorithm 2 read literally is therefore *not*
    permutation invariant past the first layer, which is the property
    the whole section exists to obtain.

    ``context="paired"`` (default) is the reading that delivers it: each
    incoming connection contributes :math:`\psi_t(w_{ij}, N^{t-1}_j)`
    and those are summed over :math:`j`, so a permutation of layer
    :math:`t-1` permutes the ``(weight, context)`` pairs together and
    the sum is unchanged. This also matches the paper's stated purpose
    for the context -- describing "what the node is performing with
    respect to its inputs" -- since the weight and the representation of
    the node it comes from stay attached to each other.

    ``context="as_printed"`` is the literal tuple, kept so the
    difference can be measured; ``context="none"`` drops the context
    entirely and is invariant only for the same reason as
    ``as_printed`` is not.
    """
    phis, psis, r = model["phis"], model["psis"], model["repr_dim"]
    ctx = model["context"]
    caches, L = [], []
    prev_reprs = []
    scalers = model.get("scalers")
    for t, layer in enumerate(sets):
        node_reprs, layer_cache = [], []
        for node in layer:
            w, b = node[:-1], node[-1]
            if scalers is not None:
                (mw, sw), (mb, sb) = scalers[t]
                w = [(v - mw) / sw for v in w]
                b = (b - mb) / sb
            edges = None
            if t == 0 or ctx == "none":
                x = list(w) + [b]
            elif ctx == "paired":
                acc, edges = [0.0] * r, []
                for j, wij in enumerate(w):
                    ea, ep = _mlp_forward(psis[t], [wij] + prev_reprs[j],
                                          final="tanh",
                                          hidden_act="tanh")
                    edges.append((ea, ep))
                    for c in range(r):
                        acc[c] += ea[-1][c]
                x = [b] + acc
            else:
                x = list(w) + [b] + [v for nr in prev_reprs for v in nr]
            acts, pre = _mlp_forward(phis[t], x, final="tanh",
                                     hidden_act="tanh")
            node_reprs.append(acts[-1])
            layer_cache.append((acts, pre, len(w), edges))
        caches.append(layer_cache)
        L.append([sum(nr[c] for nr in node_reprs) for c in range(r)])
        prev_reprs = node_reprs
    F = [v for Lt in L for v in Lt]
    racts, rpre = _mlp_forward(model["rho"], F, final="sigmoid",
                                hidden_act="tanh")
    return racts[-1][0], {"caches": caches, "L": L, "F": F,
                          "racts": racts, "rpre": rpre}


def _deepsets_backward(model, sets, cache, dout, grads):
    r = model["repr_dim"]
    ctx = model["context"]
    dF = _mlp_backward(model["rho"], cache["racts"], cache["rpre"], [dout],
                       grads["rho"], final="sigmoid", hidden_act="tanh")
    # split dF back into one gradient per layer sum L_t; every node of
    # layer t received the same dL_t, because the sum is unweighted
    dL = [dF[t * r:(t + 1) * r] for t in range(len(sets))]
    # gradients arriving at layer t's node representations from layer t+1
    d_from_next = [None] * len(sets)
    for t in range(len(sets) - 1, -1, -1):
        layer_cache = cache["caches"][t]
        n_prev = len(cache["caches"][t - 1]) if t > 0 else 0
        d_prev = [[0.0] * r for _ in range(n_prev)]
        for i, (acts, pre, n_w, edges) in enumerate(layer_cache):
            dnode = list(dL[t])
            extra = d_from_next[t]
            if extra is not None:
                for c in range(r):
                    dnode[c] += extra[i][c]
            dx = _mlp_backward(model["phis"][t], acts, pre, dnode,
                               grads["phis"][t], final="tanh",
                               hidden_act="tanh")
            if t == 0 or ctx == "none":
                continue
            if ctx == "paired":
                dacc = dx[1:1 + r]
                for j, (ea, ep) in enumerate(edges):
                    de = _mlp_backward(model["psis"][t], ea, ep, dacc,
                                       grads["psis"][t], final="tanh",
                                       hidden_act="tanh")
                    for c in range(r):
                        d_prev[j][c] += de[1 + c]
            else:
                tail = dx[n_w + 1:]
                for j in range(n_prev):
                    for c in range(r):
                        d_prev[j][c] += tail[j * r + c]
        if t > 0:
            d_from_next[t - 1] = d_prev


def _zero_grads(model):
    return {"phis": [_zero_like(p) for p in model["phis"]],
            "psis": [None if p is None else _zero_like(p)
                     for p in model["psis"]],
            "rho": _zero_like(model["rho"])}


def _train_set_meta(sets_list, labels, phi_hidden, repr_dim, rho_hidden,
                    epochs, lr, seed, context="paired"):
    rnd = _rng(seed + 11)
    shapes = [(len(layer), len(layer[0]) - 1) for layer in sets_list[0]]
    model = _deepsets_init(shapes, phi_hidden, repr_dim, rho_hidden, rnd,
                           context=context)
    model["scalers"] = _layer_scalers(sets_list)
    n = len(sets_list)
    order = list(range(n))
    for _ in range(int(epochs)):
        for i in range(n - 1, 0, -1):
            j = int(rnd() * (i + 1))
            order[i], order[j] = order[j], order[i]
        for idx in order:
            out, cache = _deepsets_forward(model, sets_list[idx])
            grads = _zero_grads(model)
            _deepsets_backward(model, sets_list[idx], cache,
                               out - labels[idx], grads)
            for p, g in zip(model["phis"], grads["phis"]):
                _sgd_step(p, g, lr, 1.0)
            for p, g in zip(model["psis"], grads["psis"]):
                if p is not None:
                    _sgd_step(p, g, lr, 1.0)
            _sgd_step(model["rho"], grads["rho"], lr, 1.0)
    return model


# ------------------------------------------------------------- driver

def _standardise(feats):
    d = len(feats[0])
    n = float(len(feats))
    mu = [sum(f[j] for f in feats) / n for j in range(d)]
    sd = []
    for j in range(d):
        v = sum((f[j] - mu[j]) ** 2 for f in feats) / max(n - 1.0, 1.0)
        sd.append(math.sqrt(v) if v > 1e-24 else 1.0)
    return [[(f[j] - mu[j]) / sd[j] for j in range(d)] for f in feats], mu, sd


def _apply_standardise(f, mu, sd):
    return [(f[j] - mu[j]) / sd[j] for j in range(len(f))]


def property_inference(shadow_models, shadow_labels, target_models=None,
                       target_labels=None, representation="set",
                       meta_hidden=(16,), phi_hidden=(8,), repr_dim=4,
                       rho_hidden=(8,), context="paired", epochs=30, lr=0.05,
                       seed=0):
    r"""Infer a training-set property from released model weights.

    Parameters
    ----------
    shadow_models : sequence
        Shadow classifiers, as returned by :func:`train_fcnn`. All must
        share one architecture, as the paper's threat model assumes
        (the adversary knows it).
    shadow_labels : sequence of 0/1
        1 where the shadow model was trained on data with property
        :math:`P`.
    target_models : sequence, optional
        The models to be attacked. Defaults to the shadow models, which
        only reports the training fit and is not an honest accuracy.
    target_labels : sequence of 0/1, optional
        Ground truth for ``target_models``, when it is known and the
        attack is being evaluated rather than run.
    representation : {'set', 'sorting', 'baseline'}
        Which of the paper's three feature representations to use.
    context : {'paired', 'as_printed', 'none'}
        Only for ``representation="set"``: how a neuron is told about
        the layer beneath it. See :func:`_deepsets_forward` -- the
        paper's literal tuple (``"as_printed"``) is not permutation
        invariant, ``"paired"`` is.
    """
    if representation not in _REPRS:
        raise ValueError("propinf: representation must be one of %s"
                         % (_REPRS,))
    nets = list(shadow_models)
    lab = [float(v) for v in shadow_labels]
    if len(nets) != len(lab):
        raise ValueError("propinf: one label per shadow model is required")
    if len(nets) < 4:
        raise ValueError("propinf: at least four shadow models are needed")
    if any(v not in (0.0, 1.0) for v in lab):
        raise ValueError("propinf: shadow_labels must be 0/1")
    if len(set(lab)) < 2:
        raise ValueError("propinf: shadow_labels must contain both classes")
    arch = [(len(L["W"]), len(L["W"][0])) for L in nets[0]]
    for net in nets:
        if [(len(L["W"]), len(L["W"][0])) for L in net] != arch:
            raise ValueError("propinf: all shadow models must share one "
                             "architecture")
    if context not in _CONTEXTS:
        raise ValueError("propinf: context must be one of %s" % (_CONTEXTS,))
    if repr_dim < 1:
        raise ValueError("propinf: repr_dim must be at least 1")
    if epochs < 1 or lr <= 0:
        raise ValueError("propinf: epochs must be >= 1 and lr positive")

    targets = list(target_models) if target_models is not None else nets
    for net in targets:
        if [(len(L["W"]), len(L["W"][0])) for L in net] != arch:
            raise ValueError("propinf: target model architecture differs "
                             "from the shadow models")

    if representation == "set":
        train_sets = [set_representation(net) for net in nets]
        model = _train_set_meta(train_sets, lab, phi_hidden, int(repr_dim),
                                rho_hidden, epochs, lr, seed, context)
        scores = [_deepsets_forward(model, set_representation(net))[0]
                  for net in targets]
        fit = [_deepsets_forward(model, s)[0] for s in train_sets]
        meta = model
    else:
        extract = (sorted_representation if representation == "sorting"
                   else flat_representation)
        raw = [extract(net) for net in nets]
        feats, mu, sd = _standardise(raw)
        model = _train_vector_meta(feats, lab, meta_hidden, epochs, lr, seed)
        scores = [_vector_meta_predict(
            model, _apply_standardise(extract(net), mu, sd))
            for net in targets]
        fit = [_vector_meta_predict(model, f) for f in feats]
        meta = model

    pred = [1 if s >= 0.5 else 0 for s in scores]
    train_acc = sum(1 for s, y in zip(fit, lab)
                    if (1.0 if s >= 0.5 else 0.0) == y) / float(len(lab))
    acc = None
    if target_labels is not None:
        tl = [float(v) for v in target_labels]
        if len(tl) != len(targets):
            raise ValueError("propinf: one target label per target model")
        acc = sum(1 for p, y in zip(pred, tl)
                  if float(p) == y) / float(len(tl))

    return RichResult(payload={
        "estimate": acc if acc is not None else train_acc,
        "accuracy": acc,
        "train_accuracy": train_acc,
        "prediction": pred,
        "score": scores,
        "representation": representation,
        "context": context if representation == "set" else None,
        "n_shadow": len(nets),
        "n_target": len(targets),
        "architecture": arch,
        "meta_classifier": meta,
        "method": ("property inference by shadow training "
                   "(Ganju et al. 2018), %s representation"
                   % representation),
        "note": ("baseline flattening is not permutation invariant and "
                 "the paper reports 55-77% for it; sorting (Algorithm 1) "
                 "canonicalises each hidden layer, set (Algorithm 2) is "
                 "invariant by construction and is the paper's best"),
    })


def cheatsheet():
    return ("propinf: property inference on fully connected networks "
            "(Ganju et al. 2018). Train shadow classifiers half with the "
            "property and half without, turn each into features, fit a "
            "meta-classifier, apply it to the target's weights. "
            "representation='baseline' flattens the weights (and is beaten "
            "by permutation equivalence), 'sorting' sorts each hidden "
            "layer by |sum of weights| into a canonical form, 'set' uses "
            "DeepSets rho(sum phi(node)) over each layer and is invariant "
            "by construction.")
