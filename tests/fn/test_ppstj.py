"""ppstj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppstj import ppstj


def test_ppstj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppstj()
