"""masss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.masss import masss


def test_masss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        masss()
