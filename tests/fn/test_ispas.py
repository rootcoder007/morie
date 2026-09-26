"""ispas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ispas import ispas


def test_ispas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ispas()
