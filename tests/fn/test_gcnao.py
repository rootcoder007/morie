"""gcnao is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcnao import gcnao


def test_gcnao_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcnao()
