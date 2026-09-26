"""krgsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgsm import krgsm


def test_krgsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgsm()
