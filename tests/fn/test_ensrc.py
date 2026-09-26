"""ensrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ensrc import ensrc


def test_ensrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ensrc()
