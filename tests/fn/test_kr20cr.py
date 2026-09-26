"""kr20cr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kr20cr import kuder_richardson_20


def test_kr20cr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kuder_richardson_20(X=None)
