"""vitbgi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vitbgi import vit_b16_init


def test_vitbgi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vit_b16_init(model=None)
