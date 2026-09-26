"""crfNlp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.crfNlp import crf_sequence


def test_crfNlp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        crf_sequence(X=None, y=None)
