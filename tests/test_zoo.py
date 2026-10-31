import hashlib
import io

import narwhals as nw
import numpy as np
import polars as pl
import pytest

from crowdcent_challenge import zoo
from crowdcent_challenge.client import ChallengeClient, CrowdCentAPIError

BASE_URL = "http://test.crowdcent.com/api"
SIGNED = "https://bucket.example/zoo"


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("CROWDCENT_API_KEY", raising=False)
    monkeypatch.delenv("CROWDCENT_API_URL", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def client():
    return ChallengeClient(api_key="test", base_url=BASE_URL)


def npz(coef, intercept):
    buffer = io.BytesIO()
    np.savez(
        buffer, coef=np.asarray(coef, float), intercept=np.asarray(intercept, float)
    )
    return buffer.getvalue()


def publish(requests_mock, name, kind, files, bad=()):
    """Serves one zoo version: manifest by both refs and each file behind a redirect."""
    ref = f"{name}@1"
    manifest = {
        "ref": ref,
        "name": name,
        "version": 1,
        "challenge": "hyperliquid-ranking",
        "data_version": "3.0",
        "kind": kind,
        "features": ["feature_a", "feature_b"],
        "outputs": ["pred_10d", "pred_30d"],
        "files": {
            key: {
                "name": filename,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
            for key, (filename, body) in files.items()
        },
    }
    payload = {"ref": ref, "challenge": "hyperliquid-ranking", "manifest": manifest}
    for each in (name, ref):
        requests_mock.get(f"{BASE_URL}/zoo/{each}/", json=payload)
    for key, (filename, body) in files.items():
        requests_mock.get(
            f"{BASE_URL}/zoo/{ref}/download/?file={key}",
            status_code=302,
            headers={"Location": f"{SIGNED}/{name}/{filename}"},
        )
        requests_mock.get(
            f"{SIGNED}/{name}/{filename}", content=b"tampered" if key in bad else body
        )
    return manifest


def downloads(requests_mock):
    return [r.url for r in requests_mock.request_history if r.url.startswith(SIGNED)]


INFERENCE = pl.DataFrame(
    {"id": ["a", "b", "c"], "feature_a": [0.1, 0.5, 0.9], "feature_b": [1.0, 0.0, 0.5]}
)


def test_list_models_filters_by_challenge_only_when_asked(client, requests_mock):
    listing = requests_mock.get(f"{BASE_URL}/zoo/", json=[{"ref": "m@1"}])

    assert client.list_models() == [{"ref": "m@1"}]
    assert client.list_models("hyperliquid-ranking") == [{"ref": "m@1"}]
    assert [r.qs for r in listing.request_history] == [
        {},
        {"challenge": ["hyperliquid-ranking"]},
    ]


def test_download_model_verifies_and_reuses_the_cached_file(client, requests_mock):
    body = npz([[1, 0], [0, 1]], [0, 0])
    publish(
        requests_mock, "hl-ridge-baseline", "linear", {"model": ("weights.npz", body)}
    )

    first = client.download_model("hl-ridge-baseline")
    second = client.download_model("hl-ridge-baseline")

    assert first == second
    assert first.read_bytes() == body
    assert first.parent.name == hashlib.sha256(body).hexdigest()
    assert len(downloads(requests_mock)) == 1


def test_download_model_refuses_bytes_that_do_not_match_the_manifest(
    client, requests_mock, tmp_path
):
    publish(
        requests_mock,
        "hl-ridge-baseline",
        "linear",
        {"model": ("weights.npz", npz([[1, 0], [0, 1]], [0, 0]))},
        bad={"model"},
    )

    with pytest.raises(CrowdCentAPIError, match="SHA-256"):
        client.download_model("hl-ridge-baseline", dest=tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_download_model_names_the_files_a_model_has(client, requests_mock):
    publish(
        requests_mock,
        "hl-lgb-baseline",
        "lightgbm",
        {"model_10d": ("model_10d.txt", b"x"), "model_30d": ("model_30d.txt", b"y")},
    )

    with pytest.raises(ValueError, match="model_10d, model_30d"):
        client.download_model("hl-lgb-baseline")


def test_linear_model_predicts_ranked_columns_for_a_submission(client, requests_mock):
    publish(
        requests_mock,
        "hl-ridge-baseline",
        "linear",
        {"model": ("weights.npz", npz([[1, 0], [0, -1]], [0.3, 0.1]))},
    )

    model = zoo.load("hl-ridge-baseline", client=client)
    predictions = model.predict(INFERENCE.to_pandas())

    assert model.features == ["feature_a", "feature_b"]
    assert predictions.to_dict(as_series=False) == {
        "id": ["a", "b", "c"],
        "pred_10d": [0.0, 0.5, 1.0],
        "pred_30d": [0.0, 1.0, 0.5],
    }


def test_a_single_row_predicts_the_middle_rank(client, requests_mock):
    publish(
        requests_mock,
        "hl-ridge-baseline",
        "linear",
        {"model": ("weights.npz", npz([[1, 0], [0, 1]], [0, 0]))},
    )

    predictions = zoo.load("hl-ridge-baseline", client=client).predict(INFERENCE[:1])

    assert predictions.row(0) == ("a", 0.5, 0.5)


def test_predict_names_missing_features(client, requests_mock):
    publish(
        requests_mock,
        "hl-ridge-baseline",
        "linear",
        {"model": ("weights.npz", npz([[1, 0], [0, 1]], [0, 0]))},
    )

    with pytest.raises(ValueError, match="feature_b"):
        zoo.load("hl-ridge-baseline", client=client).predict(
            INFERENCE.drop("feature_b")
        )


def test_xgboost_model_loads_from_ubj(client, requests_mock, tmp_path):
    xgboost = pytest.importorskip("xgboost")
    X = INFERENCE.select("feature_a", "feature_b").to_numpy()
    booster = (
        xgboost.XGBRegressor(n_estimators=5, base_score=0.5)
        .fit(X, np.column_stack([X[:, 0], 1 - X[:, 0]]))
        .get_booster()
    )
    booster.save_model(tmp_path / "model.ubj")
    publish(
        requests_mock,
        "hl-xgb-baseline",
        "xgboost",
        {"model": ("model.ubj", (tmp_path / "model.ubj").read_bytes())},
    )

    predictions = zoo.load("hl-xgb-baseline", client=client).predict(INFERENCE)

    assert predictions["pred_10d"].to_list() == [0.0, 0.5, 1.0]
    assert predictions["pred_30d"].to_list() == [1.0, 0.5, 0.0]


def test_lightgbm_model_loads_one_text_file_per_horizon(
    client, requests_mock, tmp_path
):
    lightgbm = pytest.importorskip("lightgbm")
    X = np.random.default_rng(0).uniform(size=(200, 2))
    files = {}
    for key, y in (("model_10d", X[:, 0]), ("model_30d", -X[:, 0])):
        path = tmp_path / f"{key}.txt"
        lightgbm.train(
            {"verbose": -1, "min_data_in_leaf": 5}, lightgbm.Dataset(X, y), 10
        ).save_model(path)
        files[key] = (path.name, path.read_bytes())
    publish(requests_mock, "hl-lgb-baseline", "lightgbm", files)

    predictions = zoo.load("hl-lgb-baseline", client=client).predict(INFERENCE)

    assert predictions["pred_10d"].to_list() == [0.0, 0.5, 1.0]
    assert predictions["pred_30d"].to_list() == [1.0, 0.5, 0.0]


def test_an_unknown_kind_is_refused_before_any_download(client, requests_mock):
    publish(requests_mock, "hl-transformer", "transformer", {"model": ("w.npz", b"x")})

    with pytest.raises(ValueError, match="cannot load"):
        zoo.load("hl-transformer", client=client)
    assert downloads(requests_mock) == []


def arrays(**values):
    buffer = io.BytesIO()
    np.savez(buffer, **{k: np.asarray(v, float) for k, v in values.items()})
    return buffer.getvalue()


def test_mlp_model_is_a_numpy_forward_pass_over_its_layers(client, requests_mock):
    weights = dict(
        x_mean=[0.5, 0.5],
        x_scale=[0.5, 0.5],
        w0=[[1.0, -1.0], [0.0, 1.0]],
        b0=[0.0, 0.0],
        w1=[[1.0, 0.0], [0.0, -1.0]],
        b1=[0.0, 0.0],
    )
    publish(
        requests_mock,
        "hl-mlp-baseline",
        "mlp",
        {"model": ("weights.npz", arrays(**weights))},
    )

    predictions = zoo.load("hl-mlp-baseline", client=client).predict(INFERENCE)

    # Scaled inputs a = (-.8, 0, .8) and b = (1, -1, 0). The hidden layer is relu(a) and
    # relu(b - a), so 10d = (0, 0, .8) and 30d = -(1.8, 0, 0), with ties sharing a rank.
    assert predictions["pred_10d"].to_list() == [0.25, 0.25, 1.0]
    assert predictions["pred_30d"].to_list() == [0.0, 0.75, 0.75]


def test_lstm_model_reads_the_features_as_steps_oldest_first(client, requests_mock):
    hidden = 1
    # One input per step and one unit: the input gate opens, the cell adds tanh(x) per step.
    w_input = np.zeros((1, 4 * hidden))
    w_input[0, 2] = 1.0
    b = np.array([50.0, 50.0, 0.0, 50.0])
    weights = dict(
        x_mean=[0.0, 0.0],
        x_scale=[1.0, 1.0],
        w_input=w_input,
        w_hidden=np.zeros((hidden, 4 * hidden)),
        b=b,
        w_out=[[1.0, -1.0]],
        b_out=[0.0, 0.0],
    )
    publish(
        requests_mock,
        "hl-lstm-baseline",
        "lstm",
        {"model": ("weights.npz", arrays(**weights))},
    )

    model = zoo.load("hl-lstm-baseline", client=client)
    raw = model._raw(nw.from_native(INFERENCE))

    expected = np.tanh(
        np.tanh(INFERENCE["feature_a"]) + np.tanh(INFERENCE["feature_b"])
    )
    np.testing.assert_allclose(raw[:, 0], expected.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(raw[:, 1], -expected.to_numpy(), atol=1e-12)


def test_blend_loads_each_member_and_rank_averages_them(client, requests_mock):
    publish(
        requests_mock,
        "hl-a",
        "linear",
        {"model": ("weights.npz", npz([[1, 0], [1, 0]], [0, 0]))},
    )
    publish(
        requests_mock,
        "hl-b",
        "linear",
        {"model": ("weights.npz", npz([[0, 1], [0, -1]], [0, 0]))},
    )
    manifest = publish(requests_mock, "hl-blend", "blend", {})
    manifest["members"] = [
        {"ref": "hl-a@1", "weight": 0.75},
        {"ref": "hl-b@1", "weight": 0.25},
    ]

    predictions = zoo.load("hl-blend", client=client).predict(INFERENCE)

    # a ranks feature_a (0, .5, 1); b ranks feature_b (1, 0, .5) for 10d and its reverse for 30d.
    assert predictions["pred_10d"].to_list() == [0.0, 0.5, 1.0]
    assert predictions["pred_30d"].to_list() == [0.0, 0.5, 1.0]
    assert sorted(url.rsplit("/", 2)[1] for url in downloads(requests_mock)) == [
        "hl-a",
        "hl-b",
    ]


def test_oof_reads_dated_out_of_fold_predictions(client, requests_mock):
    oof = b"id,date,pred_10d,pred_30d,fold\na,2026-01-01,0.0,1.0,1\nb,2026-01-01,1.0,0.0,5\n"
    publish(
        requests_mock,
        "hl-ridge-baseline",
        "linear",
        {
            "model": ("weights.npz", npz([[1, 0], [0, 1]], [0, 0])),
            "oof_csv": ("oof.csv", oof),
        },
    )

    frame = zoo.load("hl-ridge-baseline", client=client).oof()

    assert frame.columns == ["id", "date", "pred_10d", "pred_30d", "fold"]
    assert frame.schema["date"] == pl.Date
    assert frame["fold"].to_list() == [1, 5]
