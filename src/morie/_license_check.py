# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime license-compatibility guard for morie.

morie is AGPL-3.0-or-later. This module exposes a `check_plugin_license()`
helper that loaded plugins or downstream code can call to confirm
their license may be combined with morie's before consuming morie
internals. The guard
is **advisory** -- it warns or raises, but does not enforce at the
Python-import level (Python cannot prevent code from importing a
module once installed). For stronger guarantees see the companion
Linux Security Module userspace daemon at `daemon/morie_lsm.py`.
"""

from __future__ import annotations

import warnings

__all__ = [
    "GPL_COMPATIBLE_LICENSES",
    "check_plugin_license",
    "morie_license_metadata",
]


# Licenses whose code may be combined with AGPL-3.0-or-later code, per the
# FSF's license list (https://www.gnu.org/licenses/license-list.html) and
# GPLv3/AGPLv3 section 13. GPL-2.0-only is absent on purpose: it cannot be
# upgraded to version 3, so it is incompatible with every v3 license.
# Identifiers below match SPDX (https://spdx.org/licenses/).
GPL_COMPATIBLE_LICENSES: tuple[str, ...] = (
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "Apache-2.0",  # compatible with the version-3 licenses, not with GPL-2.0-only
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "CC0-1.0",
    "Unlicense",
    "Zlib",
)


def morie_license_metadata() -> dict[str, str]:
    """Return morie's SPDX-style license metadata."""
    return {
        "package": "morie",
        "spdx": "AGPL-3.0-or-later",
        "fsf_libre": "yes",
        "osi_approved": "yes",
        "kernel_compatible": "no (the Linux kernel is GPL-2.0-only; morie runs in userspace)",
    }


def check_plugin_license(plugin_spdx: str, *, raise_on_incompatible: bool = False) -> bool:
    """Check whether `plugin_spdx` may be combined with morie (AGPL-3.0-or-later).

    Args:
        plugin_spdx: SPDX-format license identifier of a downstream plugin
            (e.g. "MIT", "GPL-3.0-or-later", "Apache-2.0").
        raise_on_incompatible: If True, raise ValueError on incompatibility
            instead of warning.

    Returns:
        True if `plugin_spdx` is on the compatible list. "GPL-2.0-only"
        is not: it cannot move to version 3, so it cannot be combined
        with morie.

    Examples:
        >>> check_plugin_license("MIT")
        True
        >>> check_plugin_license("GPL-2.0-only", raise_on_incompatible=True)
        Traceback (most recent call last):
            ...
        ValueError: Plugin SPDX 'GPL-2.0-only' cannot be combined with morie (AGPL-3.0-or-later). See https://www.gnu.org/licenses/license-list.html

    Raises:
        ValueError: if `raise_on_incompatible=True` and license is not
            on the compatible list.
    """
    if not plugin_spdx:
        msg = "Plugin reports empty SPDX identifier -- cannot verify license compatibility."
        if raise_on_incompatible:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    ok = plugin_spdx in GPL_COMPATIBLE_LICENSES
    if not ok:
        msg = (
            f"Plugin SPDX {plugin_spdx!r} cannot be combined with morie "
            f"(AGPL-3.0-or-later). See https://www.gnu.org/licenses/license-list.html"
        )
        if raise_on_incompatible:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return ok
