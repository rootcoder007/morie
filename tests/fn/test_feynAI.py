"""feynAI is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.feynAI import ai_feynman


def test_feynAI_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ai_feynman(X=None, y=None)
