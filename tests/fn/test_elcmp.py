"""elcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elcmp import elcmp


def test_elcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elcmp()
