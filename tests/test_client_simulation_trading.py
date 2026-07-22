"""Mocked-transport tests for the 0.2.0 client surface: auth check,
simulation (capabilities/run/sweep-continuation/blend), signed-URL twins,
and trading (network routing, consent-flow error mapping)."""

import pytest

from crowdcent_challenge import ChallengeClient, ClientError

BASE_URL = "http://test.crowdcent.com/api"
TEST_SLUG = "test-challenge"
TEST_API_KEY = "test_api_key_123"


@pytest.fixture
def client():
    return ChallengeClient(TEST_SLUG, api_key=TEST_API_KEY, base_url=BASE_URL)


# --------------------------------------------------------------------- auth


def test_check_auth(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/auth/check/",
        json={"username": "dana", "allow_trading": False, "oms_access": False},
    )
    assert client.check_auth()["username"] == "dana"


# --------------------------------------------------------------- simulation


def test_get_simulator_capabilities(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/simulator/",
        json={"tier": {"points": 600.0}},
    )
    assert client.get_simulator_capabilities()["tier"]["points"] == 600.0


def test_run_simulation_body(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/simulator/run/",
        json={"stats": {"sharpe": 1.2}, "locked": []},
    )
    result = client.run_simulation(
        config={"n_long": 10},
        include=["curve", "holdings"],
        benchmark_trials=25,
    )
    assert result["stats"]["sharpe"] == 1.2
    body = requests_mock.last_request.json()
    assert body == {
        "config": {"n_long": 10},
        "include": ["curve", "holdings"],
        "benchmark_trials": 25,
    }


def test_run_simulation_with_token(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/simulator/run/", json={"stats": {}}
    )
    client.run_simulation(config_token="n_long=10&fee_bps=3.5")
    assert requests_mock.last_request.json() == {
        "config_token": "n_long=10&fee_bps=3.5"
    }


def test_run_sweep_walks_continuation(client, requests_mock):
    chunk1 = {
        "total": 3,
        "offset": 0,
        "results": [{"config_token": "a"}, {"config_token": "b"}],
        "next_offset": 2,
    }
    chunk2 = {
        "total": 3,
        "offset": 2,
        "results": [{"config_token": "c"}],
        "next_offset": None,
    }
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/simulator/sweep/",
        [{"json": chunk1}, {"json": chunk2}],
    )
    progress = []
    result = client.run_sweep(
        {"n_short": 5},
        {"n_long": [5, 10, 20]},
        on_chunk=lambda rows, total: progress.append((len(rows), total)),
    )
    assert [r["config_token"] for r in result["results"]] == ["a", "b", "c"]
    assert result["total"] == 3
    assert progress == [(2, 3), (3, 3)]
    # The second request continued from the server's next_offset.
    assert requests_mock.request_history[-1].json()["offset"] == 2


def test_run_blend(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/simulator/blend/",
        json={"stats": {}, "sleeves": []},
    )
    client.run_blend([{"config": {"n_long": 5}, "weight": 1.0}])
    assert requests_mock.last_request.json() == {
        "sleeves": [{"config": {"n_long": 5}, "weight": 1.0}]
    }


def test_signed_url_twins(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/meta_model/download/",
        status_code=302,
        headers={"Location": "https://signed.example/meta.parquet"},
    )
    assert client.get_meta_model_url() == "https://signed.example/meta.parquet"

    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/inference_data/current/download/",
        status_code=302,
        headers={"Location": "https://signed.example/inference.parquet"},
    )
    assert (
        client.get_inference_data_url("current")
        == "https://signed.example/inference.parquet"
    )

    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/inference_data/",
        json=[{"release_date": "2024-01-15"}, {"release_date": "2024-02-01"}],
    )
    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/inference_data/2024-02-01/",
        json={"release_date": "2024-02-01"},
    )
    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/inference_data/2024-02-01/download/",
        status_code=302,
        headers={"Location": "https://signed.example/latest.parquet"},
    )
    assert (
        client.get_inference_data_url("latest")
        == "https://signed.example/latest.parquet"
    )


# ------------------------------------------------------------------ trading


def test_trading_network_routing(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/mandate/",
        json={"sleeves": []},
    )
    client.get_mandate(network="mainnet")
    assert requests_mock.last_request.qs["network"] == ["mainnet"]

    requests_mock.put(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/mandate/",
        json={"sleeves": []},
    )
    client.set_mandate({"sleeves": [{"config": {}, "weight": 1}]})
    body = requests_mock.last_request.json()
    assert body["network"] == "testnet"  # client-side default: testnet

    requests_mock.get(
        f"{BASE_URL}/trading/accounts/", json={"accounts": [{"network": "testnet"}]}
    )
    assert client.get_trading_accounts() == [{"network": "testnet"}]


def test_preview_then_execute_payloads(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/rebalance/preview/",
        json={"run_id": 7, "plan_hash": "abcd1234abcd1234", "plan": {"trades": []}},
    )
    preview = client.preview_rebalance()
    assert preview["plan_hash"] == "abcd1234abcd1234"

    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/rebalance/",
        json={"id": 8, "status": "done"},
    )
    client.execute_rebalance(preview["plan_hash"])
    assert requests_mock.last_request.json() == {
        "network": "testnet",
        "plan_hash": "abcd1234abcd1234",
    }


def test_confirmation_required_maps_to_client_error(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/rebalance/",
        status_code=409,
        json={
            "error": {
                "code": "CONFIRMATION_REQUIRED",
                "message": "plan_hash is missing, stale, or does not match.",
            }
        },
    )
    with pytest.raises(ClientError, match=r"CONFIRMATION_REQUIRED"):
        client.execute_rebalance("stale_hash")


def test_flatten_two_step_payloads(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/flatten/",
        json={"plan_hash": "ffff0000ffff0000"},
    )
    client.flatten(preview=True)
    assert requests_mock.last_request.json() == {
        "network": "testnet",
        "preview": True,
    }
    client.flatten("ffff0000ffff0000")
    assert requests_mock.last_request.json() == {
        "network": "testnet",
        "plan_hash": "ffff0000ffff0000",
    }


def test_runs_and_orders_params(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/runs/", json={"runs": [1]}
    )
    assert client.list_rebalance_runs(limit=3) == [1]
    assert requests_mock.last_request.qs["limit"] == ["3"]

    requests_mock.get(
        f"{BASE_URL}/challenges/{TEST_SLUG}/trading/orders/", json={"orders": []}
    )
    client.list_orders(status="resting", network="mainnet")
    assert requests_mock.last_request.qs["status"] == ["resting"]
    assert requests_mock.last_request.qs["network"] == ["mainnet"]
