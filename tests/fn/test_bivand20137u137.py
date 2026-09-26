"""bivand20137u137 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bivand20137u137 import bivand2013_chapter_7_unnumbered_137


def test_bivand20137u137_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bivand2013_chapter_7_unnumbered_137(x=None)
