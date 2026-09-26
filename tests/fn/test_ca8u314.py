"""ca8u314 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ca8u314 import ca_chapter_8_unnumbered_314


def test_ca8u314_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ca_chapter_8_unnumbered_314(x=None)
