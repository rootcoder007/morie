"""dk3sg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3sg import dk3sg


def test_dk3sg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3sg()
