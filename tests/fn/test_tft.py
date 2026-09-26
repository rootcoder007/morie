"""tft is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tft import temporal_fusion_transformer


def test_tft_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        temporal_fusion_transformer(X=None, y=None, static_cov=None)
