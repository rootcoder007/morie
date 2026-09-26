"""fostrm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fostrm import fostrm


def test_fostrm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fostrm()
