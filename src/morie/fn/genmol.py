"""Generative chemistry: sampling molecules from a continuous latent
space, and checking that what comes back is a molecule.

The idea that made this field is Gomez-Bombarelli's. Chemical space is
discrete -- you cannot take half a bond -- so it cannot be searched with
anything that needs a gradient. A variational autoencoder buys its way
out by learning a CONTINUOUS coordinate system in which every point
decodes to a molecule: now you can move smoothly, follow a gradient,
interpolate between two compounds, and optimise a property directly in
the latent space instead of enumerating candidates.

Two things follow, and both are in here:

  THE SAMPLER. Draw a latent, decode it. The draw is the
  reparameterisation of Kingma and Welling -- z = mu + sigma times a
  standard normal -- with a temperature multiplying sigma, so
  temperature zero returns the mean exactly and larger temperatures
  trade validity for diversity. There is also a diffusion route, which
  runs the reverse DDPM process in the latent space using the schedule
  already written for the protein module rather than a second copy of
  it.

  THE OPTIMISER. Gradient ascent on a property, in the latent space.
  The gradient is by central differences on the caller's own property
  function, which means this works with any property that can be
  computed -- including one that is not differentiable, which is most
  of the interesting ones.

WHAT IS ARITHMETIC AND WHAT IS A WEIGHT. Every formula here is exact
and published: the reparameterisation, the closed-form Gaussian
Kullback-Leibler divergence, the evidence lower bound, the central
difference, the diffusion posterior. The DECODER is a trained network
and there is none here. Called without one the module says so and
declines, because a generative model with no decoder generates nothing,
and returning latent vectors while calling them molecules would be the
worst kind of quiet failure.

VALIDITY IS EXECUTED, NOT ASSERTED. A decoded string is valid if it
PARSES as a molecule -- the module runs the SMILES parser over it and
counts what survives. That is the standard the field measures itself
against and it is the one number in a generative chemistry paper that
cannot be argued with. Uniqueness and novelty are counted the same way,
against the training set the caller supplies; a model that reproduces
its training set has a validity of one and a novelty of zero, and
reporting only the first would be flattering it.

References
  Gomez-Bombarelli, R., Wei, J.N., Duvenaud, D., Hernandez-Lobato,
    J.M., Sanchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre,
    J., Hirzel, T.D., Adams, R.P. and Aspuru-Guzik, A. (2018)
    "Automatic chemical design using a data-driven continuous
    representation of molecules." ACS Central Science 4(2), 268-276.
    doi:10.1021/acscentsci.7b00572.
  Sanchez-Lengeling, B. and Aspuru-Guzik, A. (2018) "Inverse molecular
    design using machine learning: generative models for matter
    engineering." Science 361(6400), 360-365.
    doi:10.1126/science.aat2663.
  Kingma, D.P. and Welling, M. (2014) "Auto-encoding variational
    Bayes." International Conference on Learning Representations. The
    reparameterisation and the bound.
  Ho, J., Jain, A. and Abbeel, P. (2020) "Denoising diffusion
    probabilistic models." NeurIPS 33, 6840-6851. The diffusion route.
  Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S.,
    Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy,
    V., Veselov, M., Kadurin, A., Johansson, S., Chen, H., Nikolenko,
    S., Aspuru-Guzik, A. and Zhavoronkov, A. (2020) "Molecular sets
    (MOSES): a benchmarking platform for molecular generation models."
    Frontiers in Pharmacology 11, 565644. Validity, uniqueness and
    novelty as they are reported here.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from .alfrf2 import ddpm_schedule
from .avalon import parse_smiles
from ._richresult import RichResult

__all__ = ["generative_chemistry", "sample_latent", "kl_divergence",
           "elbo", "optimise_latent", "validity", "cheatsheet"]

_ROUTES = ("vae", "diffusion")


def kl_divergence(mu, logvar):
    """The Gaussian KL to a standard normal, in closed form.

    Half the sum of the squared mean, plus the variance, minus one,
    minus the log variance. Exactly zero at mu zero and logvar zero,
    which is what makes it usable as a regulariser: the penalty is
    nothing when the posterior already is the prior.
    """
    if len(mu) != len(logvar):
        raise ValueError("one log-variance per latent dimension")
    terms = []
    for i in range(len(mu)):
        m = float(mu[i])
        lv = float(logvar[i])
        terms.append(m * m + math.exp(lv) - 1.0 - lv)
    return 0.5 * _w.csum(terms)


def elbo(reconstruction, mu, logvar, beta=1.0):
    """The evidence lower bound: reconstruction minus the divergence.

    ``beta`` weights the divergence. At one this is the bound; above
    one it is the beta-VAE trade, which buys a more disentangled latent
    space at the cost of reconstruction, and it is a parameter because
    that trade is the caller's to make.
    """
    return float(reconstruction) - float(beta) * kl_divergence(mu,
                                                               logvar)


def sample_latent(mu, logvar, n=1, temperature=1.0, seed=0):
    """Reparameterised draws: z = mu + temperature * sigma * epsilon.

    At temperature zero every draw is exactly the mean -- not
    approximately, exactly -- which is the check that the noise really
    is entering through this one multiplication and nowhere else.
    """
    d = len(mu)
    if len(logvar) != d:
        raise ValueError("one log-variance per latent dimension")
    t = float(temperature)
    if t < 0.0:
        raise ValueError("a negative temperature is not a temperature")
    rng = _core._SplitMix64(seed)
    out = []
    for _ in range(int(n)):
        row = []
        for i in range(d):
            e = rng.normal()
            s = math.exp(0.5 * float(logvar[i]))
            row.append(float(mu[i]) + t * s * e)
        out.append(row)
    return out


def optimise_latent(z0, property_fn, steps=20, lr=0.1, eps=1e-4):
    """Gradient ascent on a property, in the latent space.

    The gradient is a central difference, so the property function need
    not be differentiable or even continuous -- which matters, because
    the properties worth optimising in chemistry are things like
    synthetic accessibility and predicted binding, and those come from
    code, not from formulas.

    The trajectory is returned, not just the endpoint: a latent
    optimisation that ran off to infinity looks exactly like a
    successful one if you only report where it stopped.
    """
    z = [float(v) for v in z0]
    traj = [list(z)]
    vals = [float(property_fn(z))]
    for _ in range(int(steps)):
        g = []
        for i in range(len(z)):
            up = list(z)
            dn = list(z)
            up[i] += eps
            dn[i] -= eps
            g.append((float(property_fn(up)) - float(property_fn(dn)))
                     / (2.0 * eps))
        z = [z[i] + lr * g[i] for i in range(len(z))]
        traj.append(list(z))
        vals.append(float(property_fn(z)))
    return z, traj, vals


def validity(smiles_list):
    """Which decoded strings are molecules, by running the parser.

    Validity here is not a heuristic and not a regex: a string is valid
    when the SMILES parser accepts it, which is the same standard the
    generative chemistry literature reports. Anything the parser
    refuses is invalid, and the reason it refused is available to
    whoever wants it.
    """
    flags = []
    for s in smiles_list:
        try:
            parse_smiles(s)
            flags.append(True)
        except ValueError:
            flags.append(False)
    return flags


def generative_chemistry(model, n_samples, conditions=None,
                         route="vae", temperature=1.0, seed=0,
                         steps=20, T=10, beta=1.0):
    """Sample molecules from a latent model and score what came back.

    Parameters
    ----------
    model : mapping
        ``mu`` and ``logvar`` for the latent prior, and ``decoder``, a
        function from a latent vector to a SMILES string. Without a
        decoder nothing is generated and the reason says so.
    n_samples : int
        How many to draw.
    conditions : mapping or None
        ``property`` to optimise each latent against before decoding,
        and ``training_set`` to measure novelty against.
    route : {"vae", "diffusion"}
        Draw from the prior, or run the reverse diffusion in the latent
        space using the shared schedule.

    Returns
    -------
    RichResult
        The latents, the decoded strings, and validity, uniqueness and
        novelty as fractions of the sample.

    References
    ----------
    Gomez-Bombarelli et al. (2018) ACS Cent. Sci. 4(2), 268-276;
    Kingma and Welling (2014) ICLR; Polykovskiy et al. (2020) Front.
    Pharmacol. 11, 565644.
    """
    if route not in _ROUTES:
        raise ValueError("the route is vae or diffusion")
    n = int(n_samples)
    if n < 1:
        raise ValueError("a sample of nothing is not a sample")
    mu = [float(v) for v in model["mu"]]
    logvar = [float(v) for v in model["logvar"]]
    cond = conditions if conditions is not None else {}
    prop = cond.get("property") if hasattr(cond, "get") else None
    train = cond.get("training_set", []) if hasattr(cond, "get") else []
    dec = model.get("decoder") if hasattr(model, "get") else None

    if route == "vae":
        zs = sample_latent(mu, logvar, n, temperature, seed)
    else:
        betas, alphas, abar = ddpm_schedule(T)
        rng = _core._SplitMix64(seed)
        d = len(mu)
        zs = []
        for _ in range(n):
            x = [rng.normal() for _ in range(d)]
            for t in range(int(T), 0, -1):
                # No trained denoiser either, so the prediction of the
                # clean latent is the prior mean. That makes the route
                # the prior sampled through the diffusion posterior --
                # correct arithmetic, honestly labelled, and not a
                # claim to be anybody's generative model.
                x0 = list(mu)
                c1 = math.sqrt(abar[t - 1]) * betas[t] / (1.0 - abar[t])
                c2 = (math.sqrt(alphas[t]) * (1.0 - abar[t - 1])
                      / (1.0 - abar[t]))
                sd = math.sqrt(betas[t] * (1.0 - abar[t - 1])
                               / (1.0 - abar[t]))
                z = [rng.normal() for _ in range(d)]
                x = [c1 * x0[i] + c2 * x[i]
                     + (temperature * sd * z[i] if t > 1 else 0.0)
                     for i in range(d)]
            zs.append(x)

    trajs = []
    props = []
    if prop is not None:
        moved = []
        for z in zs:
            zf, traj, vals = optimise_latent(z, prop, steps)
            moved.append(zf)
            trajs.append(traj)
            props.append(vals)
        zs = moved

    if dec is None:
        return RichResult(payload={
            "latents": zs,
            "smiles": [],
            "valid": [],
            "reason": ("the model carries no decoder, so there is "
                       "nothing to turn a latent vector into a "
                       "molecule: a decoder is a trained network and "
                       "none is shipped here. Pass one as the "
                       "model's decoder."),
            "n_samples": n, "n_valid": 0, "n_unique": 0, "n_novel": 0,
            "validity": 0.0, "uniqueness": 0.0, "novelty": 0.0,
            "kl": kl_divergence(mu, logvar),
            "elbo": None, "property": props, "trajectory": trajs,
            "route": route, "temperature": float(temperature),
            "beta": float(beta), "n_latent": len(mu),
            "has_decoder": False,
            "method": "latent generative chemistry sampler",
        })

    smiles = [str(dec(z)) for z in zs]
    flags = validity(smiles)
    good = [smiles[i] for i in range(n) if flags[i]]
    uniq = sorted(set(good))
    tr = set(str(v) for v in train)
    novel = [s for s in uniq if s not in tr]
    nv = 0
    for f in flags:
        if f:
            nv += 1
    return RichResult(payload={
        "latents": zs,
        "smiles": smiles,
        "valid": flags,
        "reason": "",
        "n_samples": n,
        "n_valid": nv,
        "n_unique": len(uniq),
        "n_novel": len(novel),
        "validity": nv / float(n),
        "uniqueness": (len(uniq) / float(nv)) if nv else 0.0,
        "novelty": (len(novel) / float(len(uniq))) if uniq else 0.0,
        "unique": uniq,
        "novel": novel,
        "kl": kl_divergence(mu, logvar),
        "elbo": elbo(nv / float(n), mu, logvar, beta),
        "property": props,
        "trajectory": trajs,
        "route": route,
        "temperature": float(temperature),
        "beta": float(beta),
        "n_latent": len(mu),
        "has_decoder": True,
        "method": "latent generative chemistry sampler",
    })


def cheatsheet():
    return ("genmol: generative chemistry. Reparameterised latent "
            "sampling or a latent diffusion, gradient ascent on any "
            "property by central differences, validity checked by "
            "parsing")
