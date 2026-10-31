"""Model zoo: list the ready-made models and download their verified files."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from ..exceptions import CrowdCentAPIError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zoo_cache_dir() -> Path:
    """The folder that keeps downloaded zoo files, outside any project folder."""
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "crowdcent" / "zoo"


class ZooAPI:
    def list_models(self, challenge: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists the live zoo models, optionally for one challenge only.

        Args:
            challenge: A challenge slug such as ``"hyperliquid-ranking"``.
                ``None`` lists the live models of every challenge.

        Returns:
            A list of dictionaries with ``ref``, ``challenge``, ``manifest``
            and ``live``.
        """
        params = {"challenge": challenge} if challenge else None
        return self._request("GET", "/zoo/", params=params).json()

    def get_model(self, ref: str) -> Dict[str, Any]:
        """Returns one zoo model version.

        Args:
            ref: ``name@version``, or a bare name for the newest live version.

        Returns:
            A dictionary with ``ref``, ``challenge``, ``manifest`` and ``live``.
        """
        return self._request("GET", f"/zoo/{quote(ref, safe='@')}/").json()

    def download_model(
        self, ref: str, file: str = "model", dest: Optional[str] = None
    ) -> Path:
        """Downloads one file of a zoo model and checks its SHA-256.

        A file already on disk with the right SHA-256 is not downloaded again.

        Args:
            ref: ``name@version``, or a bare name for the newest live version.
            file: The manifest's key for the file: ``model``, ``model_10d``,
                ``model_30d``, ``oof``, ``oof_csv`` or ``check``.
            dest: The folder to save into. Defaults to the zoo cache,
                ``~/.cache/crowdcent/zoo/<sha256>/``.

        Returns:
            The path of the verified file.

        Raises:
            ValueError: If the model has no file with that key.
            CrowdCentAPIError: If the downloaded bytes do not match the manifest.
        """
        manifest = self.get_model(ref)["manifest"]
        files = manifest["files"]
        if file not in files:
            raise ValueError(
                f"{manifest['ref']} has no '{file}' file. Use one of: {', '.join(files)}."
            )
        entry = files[file]
        folder = Path(dest) if dest else zoo_cache_dir() / entry["sha256"]
        path = folder / entry["name"]
        if path.is_file() and _sha256(path) == entry["sha256"]:
            return path
        folder.mkdir(parents=True, exist_ok=True)
        self._download_file(
            f"/zoo/{quote(manifest['ref'], safe='@')}/download/",
            str(path),
            f"{manifest['ref']} {file}",
            params={"file": file},
        )
        if _sha256(path) != entry["sha256"]:
            path.unlink()
            raise CrowdCentAPIError(
                f"{manifest['ref']} {file} does not match its SHA-256 in the manifest."
            )
        return path
