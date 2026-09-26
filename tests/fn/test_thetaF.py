"""thetaF re-exports the real theta_method from esttsl."""

from morie.fn.esttsl import theta_method as canonical
from morie.fn.thetaF import theta_method


def test_thetaF_is_the_canonical_implementation():
    assert theta_method is canonical
