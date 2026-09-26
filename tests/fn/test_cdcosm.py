"""cdcosm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdcosm import cdcosm


def test_cdcosm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdcosm()
