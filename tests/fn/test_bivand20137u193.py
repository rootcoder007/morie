"""bivand20137u193 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bivand20137u193 import bivand2013_chapter_7_unnumbered_193


def test_bivand20137u193_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bivand2013_chapter_7_unnumbered_193(x=None)
