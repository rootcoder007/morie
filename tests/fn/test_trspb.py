"""trspb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trspb import trspb


def test_trspb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trspb()
