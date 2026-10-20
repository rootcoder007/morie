# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproducible data capsules, in the format rmoriebricklayer verifies.

A capsule is a directory holding a data file pinned by a provenance record
(``data_provenance.json``), optionally a results manifest (``manifest.json``)
and a signed bundle (``capsule_bundle.json``). Everything written here is
read and verified by the R package rmoriebricklayer (``vrfcps()``,
``cpsbndlvrf()``, ``mnfrcmp()``, ``cpsdrft()``) and by the GitHub Action
``rootcoder007/rmbl-action``, and capsules written by them are verified here.

* Provenance: :func:`write`, :func:`load`, :func:`pin`, :func:`verify`.
* Manifests: :func:`manifest`, :func:`record`, :func:`write_manifest`,
  :func:`read_manifest`, :func:`canon`, :func:`digest`, :func:`compare`.
* Signing: :func:`sign`, :func:`verify_signature`, :func:`attest`,
  :func:`bundle`, :func:`read_bundle`, :func:`verify_bundle`. morie signs
  with the ``hmac`` scheme (a shared secret) in this version; bundles signed
  with ML-DSA, SLH-DSA or XMSS keys are verified by the R package and the
  action, not here.
* Drift: :func:`drift`, :func:`benford`.

The same commands are on the command line as ``morie capsule init | pin |
verify | sign | drift``.
"""

from ._drift import benford, drift
from ._manifest import canon, compare, digest, manifest, read_manifest, record, write_manifest
from ._prov import load, pin, sha256file, verify, write
from ._sign import attest, bundle, check_attestation, key_id, read_bundle, sign, verify_bundle, verify_signature

__all__ = [
    "attest",
    "benford",
    "bundle",
    "canon",
    "check_attestation",
    "compare",
    "digest",
    "drift",
    "key_id",
    "load",
    "manifest",
    "pin",
    "read_bundle",
    "read_manifest",
    "record",
    "sha256file",
    "sign",
    "verify",
    "verify_bundle",
    "verify_signature",
    "write",
    "write_manifest",
]
