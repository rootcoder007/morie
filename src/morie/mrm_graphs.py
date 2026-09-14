# SPDX-License-Identifier: AGPL-3.0-or-later
"""Causal DAGs for MRM, and the bundled starting-point structures.

Parity with rmorie's R/dag_native.R (`morie_dag`, `morie_mrm_dags`).

A DAG here is a thin, checked container over an edge list. It carries
its own acyclicity proof and keeps edges as ``(from, to)`` tuples, which
is the representation `morie.viz.dag_plot` and
`morie.fn.bdcrt.backdoor_criterion` already take -- so a graph built
here plots and identifies without conversion.

The bundled structures are starting points for justice-system work, not
claims about a particular jurisdiction. Both encode the same shape: a
set of common causes of the exposure and the outcome, which is exactly
the shape that makes an unadjusted comparison misleading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from morie.fn.bdcrt import backdoor_criterion as _backdoor_criterion

__all__ = ["CausalDag", "causal_dag", "mrm_dags"]


@dataclass
class CausalDag:
    nodes: list[str]
    edges: list[tuple[str, str]]
    exposure: str
    outcome: str
    latent: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return ("CausalDag: %d nodes, %d edges, %s -> %s%s"
                % (len(self.nodes), len(self.edges), self.exposure,
                   self.outcome,
                   "" if not self.latent
                   else " (latent: %s)" % ", ".join(self.latent)))

    def parents(self, node: str) -> list[str]:
        return [u for u, v in self.edges if v == node]

    def children(self, node: str) -> list[str]:
        return [v for u, v in self.edges if u == node]

    def backdoor(self, adjust=()):
        """Does `adjust` satisfy the back-door criterion for this graph?

        Delegates to morie.fn.bdcrt, which implements Pearl's two
        conditions; the result exposes `.satisfied`.
        """
        if isinstance(adjust, str):
            adjust = (adjust,)
        return _backdoor_criterion(self.edges, self.exposure,
                                   self.outcome, tuple(adjust))


def _parse_edge(e) -> tuple[str, str]:
    if isinstance(e, (tuple, list)):
        if len(e) != 2 or not all(str(p).strip() for p in e):
            raise ValueError("edge must be a pair of node names: %r" % (e,))
        return (str(e[0]).strip(), str(e[1]).strip())
    parts = [p.strip() for p in str(e).split("->")]
    if len(parts) != 2 or not all(parts):
        raise ValueError("edge must look like 'A -> B': %s" % e)
    return (parts[0], parts[1])


def causal_dag(edges, exposure: str, outcome: str,
               latent=()) -> CausalDag:
    """Build a DAG from edges, rejecting cycles.

    `edges` may be ``"A -> B"`` strings or ``(from, to)`` pairs. The
    exposure and the outcome must both appear in the graph: a DAG whose
    exposure is not in it cannot identify anything, and saying so here
    beats an empty adjustment set later.
    """
    if isinstance(edges, str):
        edges = [edges]
    em = [_parse_edge(e) for e in edges]
    if not em:
        raise ValueError("`edges` must contain at least one edge")
    nodes: list[str] = []
    for u, v in em:
        for n in (u, v):
            if n not in nodes:
                nodes.append(n)
    if exposure not in nodes:
        raise ValueError("exposure not in graph: %s" % exposure)
    if outcome not in nodes:
        raise ValueError("outcome not in graph: %s" % outcome)

    # Acyclicity by Kahn's algorithm, as the R side does. Counting the
    # nodes it can retire is the proof: anything left over is in a cycle.
    indeg = {n: 0 for n in nodes}
    for _, v in em:
        indeg[v] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        v = queue.pop(0)
        seen += 1
        for w in [b for a, b in em if a == v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    if seen != len(nodes):
        raise ValueError("graph has a cycle")

    if isinstance(latent, str):
        latent = [latent]
    return CausalDag(nodes=nodes, edges=em, exposure=exposure,
                     outcome=outcome, latent=list(latent))


def mrm_dags() -> dict[str, CausalDag]:
    """The bundled MRM starting-point graphs.

    `placement`: race, prior record and age are common causes of both
    the placement decision and the outcome, so comparing outcomes by
    placement without adjusting for them confounds the three.

    `use_of_force`: neighbourhood and race cause both police contact and
    force, so force conditional on contact is not the effect of contact.
    """
    return {
        "placement": causal_dag(
            [
                "race -> placement", "race -> outcome",
                "prior_record -> placement", "prior_record -> outcome",
                "age -> placement", "age -> outcome",
                "placement -> outcome",
            ],
            exposure="placement", outcome="outcome"),
        "use_of_force": causal_dag(
            [
                "neighbourhood -> police_contact",
                "neighbourhood -> force",
                "race -> police_contact", "race -> force",
                "police_contact -> force",
            ],
            exposure="police_contact", outcome="force"),
    }
