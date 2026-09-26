"""sosul is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sosul import sosul


def test_sosul_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sosul()
