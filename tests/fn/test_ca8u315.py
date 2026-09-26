"""ca8u315 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ca8u315 import ca_chapter_8_unnumbered_315


def test_ca8u315_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ca_chapter_8_unnumbered_315(x=None)
