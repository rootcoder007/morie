"""ca8u316 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ca8u316 import ca_chapter_8_unnumbered_316


def test_ca8u316_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ca_chapter_8_unnumbered_316(x=None)
