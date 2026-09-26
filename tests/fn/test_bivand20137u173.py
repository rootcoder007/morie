"""bivand20137u173 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bivand20137u173 import bivand2013_chapter_7_unnumbered_173


def test_bivand20137u173_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bivand2013_chapter_7_unnumbered_173(x=None)
