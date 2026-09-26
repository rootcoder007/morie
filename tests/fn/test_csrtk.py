"""csrtk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csrtk import csrtk


def test_csrtk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csrtk()
