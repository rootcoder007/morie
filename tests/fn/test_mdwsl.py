"""mdwsl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdwsl import mdwsl


def test_mdwsl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdwsl()
