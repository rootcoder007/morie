"""hyidf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyidf import hyidf


def test_hyidf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyidf()
