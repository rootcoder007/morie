"""fobas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fobas import fobas


def test_fobas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fobas()
