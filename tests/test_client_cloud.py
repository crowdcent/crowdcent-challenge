"""Mocked-transport tests for the CloudAPI client area: the ten Cloud
operations, idempotency headers, the base_version concurrency token, and
error-code mapping (VERSION_CONFLICT and friends surface verbatim)."""

import pytest

from crowdcent_challenge import ChallengeClient, ClientError, CrowdCentAPIError

BASE_URL = "http://test.crowdcent.com/api"
TEST_API_KEY = "test_api_key_123"


@pytest.fixture
def client():
    return ChallengeClient("any-challenge", api_key=TEST_API_KEY, base_url=BASE_URL)


def test_list_cloud_recipes(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/cloud/recipes/",
        json=[{"slug": "hello-cloud", "title": "Hello, Cloud"}],
    )
    assert client.list_cloud_recipes()[0]["slug"] == "hello-cloud"


def test_list_and_get_projects_are_account_scoped_not_challenge_scoped(
    client, requests_mock
):
    listed = requests_mock.get(f"{BASE_URL}/cloud/projects/", json=[])
    detail = requests_mock.get(
        f"{BASE_URL}/cloud/projects/abc123/",
        json={"id": "abc123", "latest_version": 3},
    )

    assert client.list_cloud_projects() == []
    assert client.get_cloud_project("abc123")["latest_version"] == 3
    # No challenge slug anywhere in the path.
    assert "any-challenge" not in listed.last_request.path
    assert "any-challenge" not in detail.last_request.path


def test_create_from_source_sends_filename_and_idempotency_header(
    client, requests_mock
):
    requests_mock.post(
        f"{BASE_URL}/cloud/projects/", json={"id": "abc123"}, status_code=201
    )

    client.create_cloud_project(
        "my-notebook",
        source="print('hi')\n",
        filename="daily.py",
        idempotency_key="create-1",
    )

    request = requests_mock.last_request
    assert request.json() == {
        "name": "my-notebook",
        "source": "print('hi')\n",
        "filename": "daily.py",
    }
    assert request.headers["Idempotency-Key"] == "create-1"


def test_create_from_recipe_sends_no_source_and_a_generated_key(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/cloud/projects/", json={"id": "abc123"}, status_code=201
    )

    client.create_cloud_project("Dashboard", recipe="numerai-dashboard")
    body = requests_mock.last_request.json()
    assert body == {"name": "Dashboard", "recipe": "numerai-dashboard"}

    # No key passed, so the client minted one: the transport retry loop
    # replays with it, making a lost-response retry safe by default.
    first = requests_mock.last_request.headers["Idempotency-Key"]
    assert first.startswith("cc-") and len(first) <= 64

    # A second *call* is a genuinely new request, not a retry: fresh key.
    client.create_cloud_project("Dashboard", recipe="numerai-dashboard")
    assert requests_mock.last_request.headers["Idempotency-Key"] != first


def test_update_sends_the_base_version_token(client, requests_mock):
    requests_mock.patch(f"{BASE_URL}/cloud/projects/abc123/", json={"id": "abc123"})

    client.update_cloud_project("abc123", files={"notebook.py": "print('v2')\n"}, base_version=1)

    assert requests_mock.last_request.json() == {
        "files": {"notebook.py": "print('v2')\n"},
        "base_version": 1,
    }


def test_a_stale_save_surfaces_version_conflict_verbatim(client, requests_mock):
    requests_mock.patch(
        f"{BASE_URL}/cloud/projects/abc123/",
        status_code=409,
        json={
            "error": {
                "code": "VERSION_CONFLICT",
                "message": "This project is at version 3, not version 1.",
            }
        },
    )

    with pytest.raises(ClientError, match="VERSION_CONFLICT"):
        client.update_cloud_project("abc123", files={"notebook.py": "x"}, base_version=1)


def test_run_pins_version_envelope_and_idempotency(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/cloud/projects/abc123/runs/",
        json={"id": "run-uuid", "state": "queued"},
        status_code=201,
    )

    run = client.run_cloud_project(
        "abc123", version=2, envelope="m", idempotency_key="run-1"
    )

    assert run["state"] == "queued"
    request = requests_mock.last_request
    assert request.json() == {"envelope": "m", "version": 2}
    assert request.headers["Idempotency-Key"] == "run-1"


def test_run_sends_its_parameters(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/cloud/projects/abc123/runs/",
        json={"id": "run-uuid", "state": "queued", "parameters": {"seed": 3}},
        status_code=201,
    )

    client.run_cloud_project("abc123", parameters={"seed": 3, "target": "30d"})

    assert requests_mock.last_request.json() == {
        "envelope": "s",
        "parameters": {"seed": 3, "target": "30d"},
    }


def test_run_without_a_key_still_sends_a_generated_one(client, requests_mock):
    requests_mock.post(
        f"{BASE_URL}/cloud/projects/abc123/runs/",
        json={"id": "run-uuid", "state": "queued"},
        status_code=201,
    )
    client.run_cloud_project("abc123")
    assert requests_mock.last_request.headers["Idempotency-Key"].startswith("cc-")


def test_get_cloud_run(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/cloud/runs/run-uuid/",
        json={"id": "run-uuid", "state": "done", "live": False},
    )
    assert client.get_cloud_run("run-uuid")["live"] is False


def test_schedule_daily_and_on_release_bodies(client, requests_mock):
    requests_mock.put(
        f"{BASE_URL}/cloud/projects/abc123/schedule/",
        json={"armed": True, "trigger": "daily"},
    )

    client.schedule_cloud_project(
        "abc123", "run-uuid", trigger="daily", daily_at="13:30"
    )
    assert requests_mock.last_request.json() == {
        "run": "run-uuid",
        "trigger": "daily",
        "daily_at": "13:30",
        "timezone": "UTC",
    }

    client.schedule_cloud_project(
        "abc123",
        "run-uuid",
        trigger="on_inference_release",
        challenge="hyperliquid-ranking",
    )
    # No clock fields ride along on a release trigger.
    assert requests_mock.last_request.json() == {
        "run": "run-uuid",
        "trigger": "on_inference_release",
        "challenge": "hyperliquid-ranking",
    }


def test_schedule_weekly_and_monthly_bodies(client, requests_mock):
    requests_mock.put(
        f"{BASE_URL}/cloud/projects/abc123/schedule/",
        json={"armed": True, "trigger": "weekly"},
    )

    client.schedule_cloud_project(
        "abc123",
        "run-uuid",
        trigger="weekly",
        daily_at="02:00",
        weekday=0,
        timezone="UTC",
    )
    assert requests_mock.last_request.json() == {
        "run": "run-uuid",
        "trigger": "weekly",
        "daily_at": "02:00",
        "timezone": "UTC",
        "weekday": 0,
    }

    client.schedule_cloud_project(
        "abc123",
        "run-uuid",
        trigger="monthly",
        daily_at="07:30",
        day=15,
    )
    assert requests_mock.last_request.json() == {
        "run": "run-uuid",
        "trigger": "monthly",
        "daily_at": "07:30",
        "timezone": "UTC",
        "day": 15,
    }


def test_pause_schedule_returns_a_readable_ack(client, requests_mock):
    requests_mock.delete(f"{BASE_URL}/cloud/projects/abc123/schedule/", status_code=204)
    assert client.pause_cloud_project_schedule("abc123") == {"paused": True}


def test_get_cloud_billing_is_a_plain_read(client, requests_mock):
    requests_mock.get(
        f"{BASE_URL}/cloud/billing/",
        json={"available_cents": 1450, "purchased": {"available_cents": 1200}},
    )
    assert client.get_cloud_billing()["available_cents"] == 1450
    assert requests_mock.last_request.method == "GET"


def test_a_folder_of_scripts_runs_by_name_and_chains(client, requests_mock):
    """Roo's repository: three scripts, one project; each runs by name with a
    time limit, and the schedule chains them in the folder's order."""
    create = requests_mock.post(f"{BASE_URL}/cloud/projects/", json={"id": "p1"}, status_code=201)
    run = requests_mock.post(f"{BASE_URL}/cloud/projects/p1/runs/", json={"id": "r1"}, status_code=201)
    schedule = requests_mock.put(f"{BASE_URL}/cloud/projects/p1/schedule/", json={"armed": True})

    client.create_cloud_project(
        "roo",
        source="print('train')",
        filename="train_and_predict.py",
        files={"download_data.py": "print('download')", "submit.py": "print('submit')"},
        idempotency_key="k1",
    )
    client.run_cloud_project(
        "p1", envelope="m", time_limit_minutes=30, entrypoint="download_data.py", idempotency_key="k2"
    )
    client.schedule_cloud_project("p1", "r1", trigger="after", after="download_data.py")

    assert create.last_request.json()["files"] == {
        "download_data.py": "print('download')",
        "submit.py": "print('submit')",
    }
    assert create.last_request.json()["filename"] == "train_and_predict.py"
    assert run.last_request.json() == {
        "envelope": "m",
        "time_limit_minutes": 30,
        "entrypoint": "download_data.py",
    }
    assert schedule.last_request.json() == {"run": "r1", "trigger": "after", "after": "download_data.py"}


def test_read_and_download_pinned_project_files(client, requests_mock, tmp_path):
    endpoint = f"{BASE_URL}/cloud/projects/p1/files/"
    requests_mock.get(endpoint, json={"file": {"path": "helper.py", "text": "VALUE = 2"}})
    assert client.get_cloud_project_files("p1", path="helper.py", version=2)["file"]["text"] == "VALUE = 2"
    assert requests_mock.last_request.qs["version"] == ["2"]
    requests_mock.get(endpoint, content=b"model bytes")
    destination = tmp_path / "model.joblib"
    client.download_cloud_project_file("p1", "models/best.joblib", str(destination), snapshot=12)
    assert destination.read_bytes() == b"model bytes"
    assert requests_mock.last_request.qs["snapshot"] == ["12"]
    assert requests_mock.last_request.qs["download"] == ["1"]


def test_delete_edits_and_single_schedule_pause(client, requests_mock):
    requests_mock.patch(f"{BASE_URL}/cloud/projects/p1/", json={"latest_version": 3})
    client.update_cloud_project("p1", base_version=2, files={"old.py": None, "predict.py": "print(1)"})
    assert requests_mock.last_request.json()["files"] == {"old.py": None, "predict.py": "print(1)"}
    assert requests_mock.last_request.json()["base_version"] == 2
    requests_mock.delete(f"{BASE_URL}/cloud/projects/p1/schedule/", status_code=204)
    client.pause_cloud_project_schedule("p1", entrypoint="predict.py")
    assert requests_mock.last_request.qs == {"entrypoint": ["predict.py"]}


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_model_download_keeps_existing_file(client, tmp_path, monkeypatch, interrupted):
    import requests

    destination = tmp_path / "best.joblib"
    destination.write_bytes(b"previous model")

    class Response:
        headers = {}
        closed = False

        def iter_content(self, chunk_size):
            yield b"new model first chunk"
            if interrupted:
                raise requests.exceptions.ChunkedEncodingError("connection lost")

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises(CrowdCentAPIError):
        client.download_cloud_project_file("p1", "models/best.joblib", str(destination), sha256="0" * 64)
    assert destination.read_bytes() == b"previous model"
    assert response.closed
    assert list(tmp_path.iterdir()) == [destination]
