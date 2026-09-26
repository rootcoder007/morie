"""cwwsym is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cwwsym import cwt_morlet


def test_cwwsym_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cwt_morlet(y=None, scales=None)
