"""foshan is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foshan import foshan


def test_foshan_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foshan()
