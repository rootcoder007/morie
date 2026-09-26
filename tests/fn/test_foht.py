"""foht is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foht import foht


def test_foht_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foht()
