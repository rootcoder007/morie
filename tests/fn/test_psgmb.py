"""psgmb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psgmb import psgmb


def test_psgmb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        psgmb()
