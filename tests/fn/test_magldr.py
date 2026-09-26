"""magldr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.magldr import magldr


def test_magldr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        magldr()
