"""ghsdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghsdm import ghsdm


def test_ghsdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghsdm()
