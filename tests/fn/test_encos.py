"""encos is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.encos import encos


def test_encos_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        encos()
