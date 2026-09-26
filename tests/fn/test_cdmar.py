"""cdmar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdmar import cdmar


def test_cdmar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdmar()
