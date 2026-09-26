"""guide_on_data_analysis1e10 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.guide_on_data_analysis1e10 import guide_on_data_analysis_chapter_1_equation_10


def test_guide_on_data_analysis1e10_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        guide_on_data_analysis_chapter_1_equation_10(x=None)
