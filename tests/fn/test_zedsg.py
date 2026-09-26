"""zedsg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zedsg import disease_map_gamma


def test_zedsg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        disease_map_gamma(data=None)
