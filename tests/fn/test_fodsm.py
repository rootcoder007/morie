"""fodsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fodsm import fodsm


def test_fodsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fodsm()
