"""aitbi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitbi import aitchison_biplot


def test_aitbi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aitchison_biplot(X=None)
