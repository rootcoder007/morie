"""btrnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.btrnd import boot_rng_seeded


def test_btrnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        boot_rng_seeded(seed=None)
