"""ca8u313 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ca8u313 import ca_chapter_8_unnumbered_313


def test_ca8u313_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ca_chapter_8_unnumbered_313(x=None)
