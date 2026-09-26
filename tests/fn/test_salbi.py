"""salbi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.salbi import salbi


def test_salbi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        salbi()
