"""zedsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zedsm import disease_map_pois


def test_zedsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        disease_map_pois(data=None)
