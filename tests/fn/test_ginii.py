"""ginii is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ginii import gini_impurity


def test_ginii_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gini_impurity(class_probs=None)
