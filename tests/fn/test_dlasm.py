"""dlasm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dlasm import dlasm


def test_dlasm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dlasm()
