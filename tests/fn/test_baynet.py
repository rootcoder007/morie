"""Tests for baynet.bayes_network."""

from morie.fn import _array_core as np

from morie.fn.baynet import bayes_network


def test_baynet_basic():
    """Test basic functionality with the classic rain/sprinkler/grass network."""
    # Rain -> Sprinkler -> Grass ; Rain -> Grass
    # Nodes: "rain", "sprinkler", "grass_wet"
    graph = {
        "rain": [],
        "sprinkler": ["rain"],
        "grass_wet": ["rain", "sprinkler"],
    }
    # CPTs as nested lists, axes (parent_1, ..., parent_k, node)
    # P(rain): [0.8, 0.2]
    # P(sprinkler | rain): dry->[0.6,0.4], wet->[0.2,0.8]
    # P(grass_wet | rain, sprinkler):
        # (rain=dry, spr=off)->[1.0,0.0]
        # (rain=dry, spr=on) ->[0.2,0.8]
        # (rain=wet,  spr=off)->[0.4,0.6]
        # (rain=wet,  spr=on) ->[0.0,1.0]
    cpts = {
        "rain": [0.8, 0.2],
        "sprinkler": [[0.6, 0.4], [0.2, 0.8]],
        "grass_wet": [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.4, 0.6], [0.0, 1.0]],
        ],
    }
    evidence = {"grass_wet": 1}  # observed wet
    query = "rain"

    result = bayes_network(graph, cpts, evidence, query)

    # Result must be a dict-like (mapping) with documented keys
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result

    # Posterior must be a proper probability distribution over rain states
    posterior = result["posterior"]
    assert len(posterior) == 2
    s = sum(posterior)
    assert abs(s - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in posterior)

    # Normalizer equals P(grass_wet=1) under the prior (no evidence).
    # P(grass_wet=1) = sum_r sum_s P(r) P(s|r) P(g=1|r,s)
    # Using the formula directly:
    normalizer_expected = 0.0
    rain_prior = cpts["rain"]
    spr_cpt = cpts["sprinkler"]        # shape [rain, spr]
    grass_cpt = cpts["grass_wet"]      # shape [rain, spr, grass]
    for r in range(2):
        for s in range(2):
            normalizer_expected += (
                rain_prior[r]
                * spr_cpt[r][s]
                * grass_cpt[r][s][1]   # grass_wet = 1
            )
    assert abs(result["normalizer"] - normalizer_expected) < 1e-9

    # And the posterior entries are joint / normalizer:
    post_expected = []
    for r in range(2):
        joint = 0.0
        for s in range(2):
            joint += rain_prior[r] * spr_cpt[r][s] * grass_cpt[r][s][1]
        post_expected.append(joint / normalizer_expected)
    assert abs(posterior[0] - post_expected[0]) < 1e-9
    assert abs(posterior[1] - post_expected[1]) < 1e-9

    # estimate must be an int in range
    est = result["estimate"]
    assert isinstance(est, int)
    assert 0 <= est < result["states"]


def test_baynet_edge():
    """Test edge case: query with no evidence yields prior over query."""
    graph = {
        "rain": [],
        "sprinkler": ["rain"],
        "grass_wet": ["rain", "sprinkler"],
    }
    cpts = {
        "rain": [0.7, 0.3],
        "sprinkler": [[0.6, 0.4], [0.2, 0.8]],
        "grass_wet": [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.4, 0.6], [0.0, 1.0]],
        ],
    }
    evidence = None
    query = "rain"

    result = bayes_network(graph, cpts, evidence, query)

    assert isinstance(result, dict)

    # With no evidence, posterior over the query equals its prior,
    # and the normalizer equals 1.0 (probability of empty evidence).
    posterior = result["posterior"]
    assert abs(posterior[0] - cpts["rain"][0]) < 1e-9
    assert abs(posterior[1] - cpts["rain"][1]) < 1e-9
    assert abs(result["normalizer"] - 1.0) < 1e-9
    assert result["estimate"] == (0 if posterior[0] >= posterior[1] else 1)
