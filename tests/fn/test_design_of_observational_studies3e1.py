"""design_of_observational_studies3e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.design_of_observational_studies3e1 import design_of_observational_studies_chapter_3_equation_1


def test_design_of_observational_studies3e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        design_of_observational_studies_chapter_3_equation_1(x=None)
