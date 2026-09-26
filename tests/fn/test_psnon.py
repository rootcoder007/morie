"""psnon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psnon import psnon


def test_psnon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        psnon()
