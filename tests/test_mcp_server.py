"""MCP server tests over fastmcp's in-memory client — no subprocess.

Covers auth-check-based trading-tool visibility, mode-aware tool
registration, pass-through to a mocked ChallengeClient, verbatim error
surfacing, base-install safety, and the hosted TokenVerifier.
"""

import subprocess
import sys
from unittest.mock import MagicMock

import narwhals as nw
import pytest

from crowdcent_challenge.mcp_server import runtime
from crowdcent_challenge.mcp_server.runtime import (
    CLOUD_TOOL_NAMES,
    TRADING_TOOL_NAMES,
)

fastmcp = pytest.importorskip("fastmcp")
from fastmcp import Client  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

from crowdcent_challenge.mcp_server.app import build_server  # noqa: E402
from crowdcent_challenge.mcp_server.hosted import CrowdCentTokenVerifier  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CROWDCENT_MCP_MODE", raising=False)
    monkeypatch.setenv("CROWDCENT_API_KEY", "test_key")
    monkeypatch.setenv("CROWDCENT_API_URL", "http://api.test/api")
    # Default: no capabilities without an explicit auth/check mock.
    runtime._stdio_claims_cache = ("test_key", float("inf"), {})


async def _tool_names(server):
    async with Client(server) as client:
        return {tool.name for tool in await client.list_tools()}


# ------------------------------------------------------------ trading gates


def _mock_auth_check(
    requests_mock, *, allow_trading=False, oms_access=False, allow_cloud=False
):
    requests_mock.get(
        "http://api.test/api/auth/check/",
        json={
            "username": "dana",
            "allow_trading": allow_trading,
            "oms_access": oms_access,
            "allow_cloud": allow_cloud,
        },
    )


async def test_trading_tools_hidden_without_capability(requests_mock):
    runtime._stdio_claims_cache = None
    _mock_auth_check(requests_mock)
    names = await _tool_names(build_server())
    assert not (names & TRADING_TOOL_NAMES)


async def test_trading_tools_visible_with_capability(requests_mock):
    runtime._stdio_claims_cache = None
    _mock_auth_check(requests_mock, allow_trading=True, oms_access=True)
    names = await _tool_names(build_server())
    assert TRADING_TOOL_NAMES <= names


async def test_stdio_visibility_uses_check_auth(requests_mock):
    runtime._stdio_claims_cache = None
    _mock_auth_check(requests_mock, allow_trading=True, oms_access=True)
    names = await _tool_names(build_server())
    assert TRADING_TOOL_NAMES <= names
    assert requests_mock.call_count == 1


async def test_hosted_visibility_filters_per_request(monkeypatch):
    monkeypatch.setenv("CROWDCENT_MCP_MODE", "hosted")
    server = build_server()
    # No verified token in the in-memory transport -> no trading capability.
    names = await _tool_names(server)
    assert not (names & TRADING_TOOL_NAMES)
    # A trade-enabled key sees them (capability check stubbed).
    monkeypatch.setattr(runtime, "request_allows_trading", lambda: True)
    names = await _tool_names(server)
    assert TRADING_TOOL_NAMES <= names


# ------------------------------------------------------------- cloud gates


async def test_cloud_tools_hidden_without_the_key_flag(requests_mock):
    runtime._stdio_claims_cache = None
    _mock_auth_check(requests_mock)
    names = await _tool_names(build_server())
    assert not (names & CLOUD_TOOL_NAMES)


async def test_cloud_tools_visible_with_the_key_flag(requests_mock):
    runtime._stdio_claims_cache = None
    _mock_auth_check(requests_mock, allow_cloud=True)
    names = await _tool_names(build_server())
    assert CLOUD_TOOL_NAMES <= names
    # The flag is cloud-only: trading tools stay hidden.
    assert not (names & TRADING_TOOL_NAMES)


async def test_one_auth_check_answers_both_visibility_filters(requests_mock):
    runtime._stdio_claims_cache = None
    _mock_auth_check(
        requests_mock, allow_trading=True, oms_access=True, allow_cloud=True
    )
    names = await _tool_names(build_server())
    assert CLOUD_TOOL_NAMES <= names
    assert TRADING_TOOL_NAMES <= names
    assert requests_mock.call_count == 1


async def test_hosted_cloud_visibility_filters_per_request(monkeypatch):
    monkeypatch.setenv("CROWDCENT_MCP_MODE", "hosted")
    server = build_server()
    names = await _tool_names(server)
    assert not (names & CLOUD_TOOL_NAMES)
    monkeypatch.setattr(runtime, "request_allows_cloud", lambda: True)
    names = await _tool_names(server)
    assert (CLOUD_TOOL_NAMES - LOCAL_FS_TOOLS) <= names


# ------------------------------------------------------------ mode awareness

LOCAL_FS_TOOLS = {
    "download_cloud_project_file",
    "download_training_dataset",
    "download_inference_data",
    "download_meta_model",
    "submit_predictions_from_file",
}
URL_TWINS = {
    "get_training_dataset_url",
    "get_inference_data_url",
    "get_meta_model_url",
}


async def test_stdio_has_file_tools_and_url_twins(monkeypatch):
    monkeypatch.setattr(runtime, "request_allows_cloud", lambda: True)
    names = await _tool_names(build_server())
    assert LOCAL_FS_TOOLS <= names
    assert URL_TWINS <= names


async def test_hosted_drops_file_tools_keeps_url_twins(monkeypatch):
    monkeypatch.setenv("CROWDCENT_MCP_MODE", "hosted")
    monkeypatch.setattr(runtime, "request_allows_cloud", lambda: True)
    names = await _tool_names(build_server())
    assert not (names & LOCAL_FS_TOOLS)
    assert URL_TWINS <= names
    assert "submit_predictions_from_dataframe" in names  # data travels inline


async def test_hosted_cannot_invoke_cloud_download_or_write_server_files(monkeypatch, tmp_path):
    from crowdcent_challenge.mcp_server import tools_cloud

    monkeypatch.setenv("CROWDCENT_MCP_MODE", "hosted")
    monkeypatch.setattr(runtime, "request_allows_cloud", lambda: True)
    fake = MagicMock()
    monkeypatch.setattr(tools_cloud, "client_for", fake)
    destination = tmp_path / "models" / "best.joblib"
    async with Client(build_server()) as client:
        with pytest.raises(ToolError, match="[Uu]nknown tool|[Nn]ot found"):
            await client.call_tool("download_cloud_project_file", {
                "project_id": "abc123", "path": "models/best.joblib", "dest_path": str(destination),
            })
    fake.assert_not_called()
    assert not destination.parent.exists()


def test_api_key_resolution_stdio(monkeypatch):
    monkeypatch.setenv("CROWDCENT_API_KEY", "stdio_key")
    assert runtime.api_key_for_request() == "stdio_key"
    client = runtime.client_for("some-challenge")
    assert client.api_key == "stdio_key"
    assert client.challenge_slug == "some-challenge"


def test_hosted_client_for_rejects_missing_bearer(monkeypatch):
    from crowdcent_challenge.exceptions import AuthenticationError

    monkeypatch.setenv("CROWDCENT_MCP_MODE", "hosted")
    monkeypatch.setenv("CROWDCENT_API_KEY", "server_key")
    with pytest.raises(AuthenticationError, match="Bearer token required"):
        runtime.client_for()


# ------------------------------------------------- pass-through + errors


async def test_run_simulation_passes_through(monkeypatch):
    from crowdcent_challenge.mcp_server import tools_simulation

    fake = MagicMock()
    fake.run_simulation.return_value = {
        "config": {"optimizer": "equal"},
        "locked": ["optimizer"],
        "stats": {"sharpe": 1.0},
    }
    monkeypatch.setattr(tools_simulation, "client_for", lambda slug: fake)
    server = build_server()
    async with Client(server) as client:
        result = await client.call_tool(
            "run_simulation",
            {"config": {"n_long": 5, "optimizer": "hrp"}, "include_curve": True},
        )
    fake.run_simulation.assert_called_once_with(
        config={"n_long": 5, "optimizer": "hrp"},
        include=["curve"],
        benchmark_trials=0,
        leverage=1.0,
        target_vol=0.0,
    )
    # Clamp echo and locked list pass through untouched.
    assert result.data["locked"] == ["optimizer"]
    assert result.data["config"]["optimizer"] == "equal"


async def test_submit_predictions_from_dataframe_uses_narwhals(monkeypatch):
    from crowdcent_challenge.mcp_server import tools_challenge

    fake = MagicMock()
    fake.submit_predictions.return_value = {"id": 42, "status": "created"}
    monkeypatch.setattr(tools_challenge, "client_for", lambda slug: fake)
    server = build_server()
    payload = '{"id": ["btc"], "pred_10d": [0.1], "pred_30d": [0.2]}'
    async with Client(server) as client:
        result = await client.call_tool(
            "submit_predictions_from_dataframe",
            {"df": payload, "slot": 2},
        )
    fake.submit_predictions.assert_called_once()
    frame = fake.submit_predictions.call_args.kwargs["df"]
    assert isinstance(frame, nw.DataFrame)
    assert list(frame.columns) == ["id", "pred_10d", "pred_30d"]
    assert result.data["id"] == 42


async def test_execute_rebalance_surfaces_errors_verbatim(monkeypatch):
    """Tool bodies have no error handling on purpose: client exceptions
    propagate and fastmcp returns their messages verbatim as tool errors."""
    from crowdcent_challenge import ClientError
    from crowdcent_challenge.mcp_server import tools_trading

    fake = MagicMock()
    fake.execute_rebalance.side_effect = ClientError(
        "Client error (409): plan_hash is missing, stale (>10 min), or does "
        "not match the newest preview. [CONFIRMATION_REQUIRED]"
    )
    monkeypatch.setattr(tools_trading, "client_for", lambda slug: fake)
    server = build_server()
    async with Client(server) as client:
        with pytest.raises(ToolError, match="CONFIRMATION_REQUIRED"):
            await client.call_tool("execute_rebalance", {"plan_hash": "deadbeef"})

    fake.execute_rebalance.side_effect = ClientError(
        "Client error (409): Another OMS action is already running for this "
        "account. Do not retry immediately. [ACCOUNT_BUSY]"
    )
    async with Client(server) as client:
        with pytest.raises(ToolError, match="ACCOUNT_BUSY"):
            await client.call_tool("execute_rebalance", {"plan_hash": "deadbeef"})


async def test_cloud_tools_pass_through_and_surface_conflicts(monkeypatch):
    from crowdcent_challenge import ClientError
    from crowdcent_challenge.mcp_server import tools_cloud

    runtime._stdio_claims_cache = ("test_key", float("inf"), {"allow_cloud": True})
    fake = MagicMock()
    fake.run_cloud_project.return_value = {"id": "run-uuid", "state": "queued"}
    fake.update_cloud_project.side_effect = ClientError(
        "Client error (409): This project is at version 3, not version 1. "
        "[VERSION_CONFLICT]"
    )
    monkeypatch.setattr(tools_cloud, "client_for", lambda: fake)
    server = build_server()

    async with Client(server) as client:
        result = await client.call_tool(
            "run_cloud_project",
            {"project_id": "abc123", "envelope": "m", "idempotency_key": "run-1"},
        )
        with pytest.raises(ToolError, match="VERSION_CONFLICT"):
            await client.call_tool(
                "update_cloud_project",
                {"project_id": "abc123", "files": {"notebook.py": "x"}, "base_version": 1},
            )

    fake.run_cloud_project.assert_called_once_with(
        "abc123",
        version=None,
        envelope="m",
        time_limit_minutes=None,
        entrypoint="",
        publish_store=None,
        idempotency_key="run-1",
        parameters=None,
    )
    assert result.data["state"] == "queued"


async def test_cloud_settings_and_archive_use_existing_rest_contract(requests_mock):
    runtime._stdio_claims_cache = ("test_key", float("inf"), {"allow_cloud": True})
    patch = requests_mock.patch("http://api.test/api/cloud/projects/abc123/", json={"id": "abc123"})
    archive = requests_mock.delete("http://api.test/api/cloud/projects/abc123/", status_code=204)
    async with Client(build_server()) as client:
        await client.call_tool("update_cloud_project", {
            "project_id": "abc123", "store_project": "models", "share_store": False,
            "publish_store": False, "base_version": 7,
            "files": {"old.py": None, "predict.py": "print('ready')\n"},
        })
        result = await client.call_tool("archive_cloud_project", {"project_id": "abc123"})
    assert patch.last_request.json() == {
        "store_project": "models", "share_store": False, "publish_store": False,
        "base_version": 7, "files": {"old.py": None, "predict.py": "print('ready')\n"},
    }
    assert archive.call_count == 1
    assert result.data == {"archived": True}


async def test_local_cloud_download_streams_pinned_bytes_to_requested_path(requests_mock, tmp_path):
    import hashlib

    runtime._stdio_claims_cache = ("test_key", float("inf"), {"allow_cloud": True})
    body = b"saved model bytes"
    digest = hashlib.sha256(body).hexdigest()
    request = requests_mock.get("http://api.test/api/cloud/projects/abc123/files/", content=body)
    destination = tmp_path / "models" / "best.joblib"
    async with Client(build_server()) as client:
        result = await client.call_tool("download_cloud_project_file", {
            "project_id": "abc123", "path": "models/best.joblib", "dest_path": str(destination),
            "snapshot": 12, "sha256": digest,
        })
    assert destination.read_bytes() == body
    assert request.last_request.qs == {
        "path": ["models/best.joblib"], "snapshot": ["12"], "sha256": [digest], "download": ["1"],
    }
    assert str(destination) in result.data
    assert list(destination.parent.iterdir()) == [destination]


# ------------------------------------------------------- base-install safety


def test_base_import_never_imports_fastmcp():
    code = (
        "import sys; import crowdcent_challenge; "
        "assert 'fastmcp' not in sys.modules, 'fastmcp leaked into base import'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_main_without_extra_gives_install_hint():
    code = (
        "import sys; sys.modules['fastmcp'] = None\n"
        "from crowdcent_challenge.mcp_server.app import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit as e:\n"
        "    assert 'crowdcent-challenge[mcp]' in str(e), str(e)\n"
        "else:\n"
        "    raise AssertionError('expected SystemExit')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------- hosted auth


async def test_token_verifier_accepts_and_caches(requests_mock):
    verifier = CrowdCentTokenVerifier(api_base="http://api.test/api")
    requests_mock.get(
        "http://api.test/api/auth/check/",
        json={"username": "dana", "allow_trading": True, "oms_access": True},
    )
    token = await verifier.verify_token("good_key")
    assert token is not None
    assert token.client_id == "dana"
    assert token.claims["allow_trading"] is True
    assert "trading" in token.scopes
    # Second call hits the cache, not the API.
    await verifier.verify_token("good_key")
    assert requests_mock.call_count == 1
    assert requests_mock.last_request.headers["Authorization"] == "Api-Key good_key"


async def test_token_verifier_rejects_bad_key(requests_mock):
    verifier = CrowdCentTokenVerifier(api_base="http://api.test/api")
    requests_mock.get(
        "http://api.test/api/auth/check/",
        status_code=403,
        json={"error": {"code": "AUTHENTICATION_FAILED", "message": "bad"}},
    )
    assert await verifier.verify_token("bad_key") is None


async def test_read_key_gets_no_trading_scope(requests_mock):
    verifier = CrowdCentTokenVerifier(api_base="http://api.test/api")
    requests_mock.get(
        "http://api.test/api/auth/check/",
        json={"username": "dana", "allow_trading": False, "oms_access": False},
    )
    token = await verifier.verify_token("read_key")
    assert token is not None
    assert token.scopes == []
