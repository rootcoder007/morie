"""msemb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msemb import embedding_qual


def test_msemb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        embedding_qual(data=None)
