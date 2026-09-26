"""dk3an is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3an import dk3an


def test_dk3an_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3an()
