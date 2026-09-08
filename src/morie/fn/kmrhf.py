# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The full RLHF pipeline: SFT -> reward model -> PPO."""

from ._richresult import RichResult

__all__ = ["kamath_rlhf_pipeline"]


def kamath_rlhf_pipeline(demos, preferences, pi0, sft=None, train_rm=None,
                         ppo=None):
    """pi_SFT = SFT(pi_0, demos); r_phi = train_RM(preferences);
    pi_RLHF = PPO(pi_SFT, r_phi, KL to pi_SFT).

    Orchestration only -- the three learned stages are the caller's
    callables, and this module enforces the wiring the pipeline is
    actually about:

    * PPO is initialised from ``pi_SFT``, not from ``pi_0``;
    * the KL reference is ``pi_SFT``, the SAME object -- passing the
      base model or a fresh copy is the classic silent bug, so the
      identity is checked and reported;
    * empty demonstrations or preferences are refused, because a
      reward model fitted to nothing still returns numbers.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, the RLHF pipeline.

    Examples
    --------
    >>> out = kamath_rlhf_pipeline(
    ...     ["demo"], [("a", "b")], "pi0",
    ...     sft=lambda p, d: p + "+sft",
    ...     train_rm=lambda prefs: (lambda y: len(y)),
    ...     ppo=lambda pi, rm, ref: pi + "+ppo")
    >>> out["policy"]
    'pi0+sft+ppo'
    >>> out["kl_reference_is_sft"]
    True
    >>> out["stages"]
    ['sft', 'reward_model', 'ppo']
    """
    for name, f in (("sft", sft), ("train_rm", train_rm), ("ppo", ppo)):
        if f is None or not callable(f):
            raise ValueError(
                f"{name} must be supplied as a callable; this module "
                "wires the RLHF stages together, it does not invent "
                "them.")
    demos = list(demos)
    prefs = list(preferences)
    if not demos:
        raise ValueError("no demonstrations; there is nothing to fine-tune on.")
    if not prefs:
        raise ValueError(
            "no preference pairs; a reward model fitted to nothing "
            "still returns numbers, which is worse than failing.")

    pi_sft = sft(pi0, demos)
    if pi_sft is None:
        raise ValueError("sft returned no policy.")
    r_phi = train_rm(prefs)
    if not callable(r_phi):
        raise ValueError(
            "train_rm must return a callable reward model r_phi(y).")
    ref = pi_sft
    pi_rlhf = ppo(pi_sft, r_phi, ref)
    if pi_rlhf is None:
        raise ValueError("ppo returned no policy.")
    return RichResult(payload={
        "policy": pi_rlhf, "policy_sft": pi_sft, "reward_model": r_phi,
        "kl_reference": ref,
        "kl_reference_is_sft": ref is pi_sft,
        "stages": ["sft", "reward_model", "ppo"],
        "n_demos": len(demos), "n_preferences": len(prefs),
        "estimate": 3, "n": 3,
        "method": "RLHF pipeline SFT -> RM -> PPO (KL anchored to SFT)"})


def cheatsheet():
    return "kmrhf: SFT -> RM -> PPO; PPO starts at and is KL-anchored to SFT"
