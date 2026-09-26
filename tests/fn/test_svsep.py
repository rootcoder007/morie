"""svsep is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svsep import separating_hyp


def test_svsep_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        separating_hyp(data=None)
