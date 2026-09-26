"""srmix is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srmix import srmix


def test_srmix_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srmix()
