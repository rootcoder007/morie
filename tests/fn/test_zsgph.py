"""zsgph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgph import gp_hyperparams


def test_zsgph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gp_hyperparams(data=None)
