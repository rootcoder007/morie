"""dtdag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtdag import dtdag


def test_dtdag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtdag()
