"""Tests for morie.fn.ddpgc: equals hmddpg.geron_ddpg."""

import inspect

from morie.fn.ddpgc import ddpg
from morie.fn.hmddpg import geron_ddpg


def test_forwards_every_argument():
    p = inspect.signature(ddpg).parameters
    g = inspect.signature(geron_ddpg).parameters
    for k in ("tau", "epochs", "lr", "gamma", "ou_theta", "ou_sigma", "seed", "s0"):
        assert k in p and k in g
        assert p[k].default == g[k].default
