"""Installing the pinned Codex CLI on the box: verified bytes in, refusals out."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from tests.serve.conftest import request
from tick.serve import codex_install
from tick.serve.codex_install import CodexInstallError, PinnedAsset, install_codex


def _tarball(
    name: str = "codex-x86_64-unknown-linux-musl", body: bytes = b"#!/bin/sh\necho codex\n"
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    archives = {
        "codex-test.tar.gz": _tarball(),
        "host-test.tar.gz": _tarball(
            "codex-code-mode-host-x86_64-unknown-linux-musl",
            b"#!/bin/sh\necho host\n",
        ),
    }
    monkeypatch.setattr(
        codex_install,
        "CODEX_ASSETS",
        {
            "x86_64": (
                PinnedAsset(
                    archive="codex-test.tar.gz",
                    sha256=hashlib.sha256(archives["codex-test.tar.gz"]).hexdigest(),
                    member="codex-x86_64-unknown-linux-musl",
                    destination="codex",
                ),
                PinnedAsset(
                    archive="host-test.tar.gz",
                    sha256=hashlib.sha256(archives["host-test.tar.gz"]).hexdigest(),
                    member="codex-code-mode-host-x86_64-unknown-linux-musl",
                    destination="codex-code-mode-host",
                ),
            )
        },
    )
    return archives


def test_pinned_asset_table_names_both_release_files_with_sha256() -> None:
    assets = codex_install.CODEX_ASSETS["x86_64"]
    assert {asset.destination for asset in assets} == {"codex", "codex-code-mode-host"}
    assert {asset.member for asset in assets} == {
        "codex-x86_64-unknown-linux-musl",
        "codex-code-mode-host-x86_64-unknown-linux-musl",
    }
    assert all(
        len(asset.sha256) == 64
        and all(character in "0123456789abcdef" for character in asset.sha256)
        for asset in assets
    )


def test_install_places_both_executables_under_home_bin(
    tmp_path: Path, pinned: dict[str, bytes]
) -> None:
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return pinned[url.rsplit("/", 1)[-1]]

    result = install_codex(tmp_path, fetch=fetch, machine="x86_64", system="Linux")

    binary = tmp_path / "bin" / "codex"
    host = tmp_path / "bin" / "codex-code-mode-host"
    assert result["code"] == "CODEX_INSTALLED" and result["path"] == str(binary)
    assert binary.read_bytes().startswith(b"#!/bin/sh")
    assert binary.stat().st_mode & 0o111 == 0o111
    assert host.read_bytes().endswith(b"echo host\n")
    assert host.stat().st_mode & 0o111 == 0o111
    assert result["paths"] == {"codex": str(binary), "codex-code-mode-host": str(host)}
    assert urls == [
        f"https://github.com/openai/codex/releases/download/{codex_install.CODEX_RELEASE_TAG}"
        "/codex-test.tar.gz",
        f"https://github.com/openai/codex/releases/download/{codex_install.CODEX_RELEASE_TAG}"
        "/host-test.tar.gz",
    ]
    assert result["reason"].endswith("You can start device login.")


def test_checksum_mismatch_installs_nothing(tmp_path: Path, pinned: dict[str, bytes]) -> None:
    with pytest.raises(CodexInstallError) as error:
        install_codex(
            tmp_path,
            fetch=lambda url: pinned[url.rsplit("/", 1)[-1]] + b"x",
            machine="x86_64",
            system="Linux",
        )
    assert error.value.code == "CODEX_CHECKSUM_MISMATCH"
    assert not (tmp_path / "bin").exists()


def test_unsupported_platform_refuses_with_the_next_step(
    tmp_path: Path, pinned: dict[str, bytes]
) -> None:
    with pytest.raises(CodexInstallError) as error:
        install_codex(
            tmp_path,
            fetch=lambda url: pinned[url.rsplit("/", 1)[-1]],
            machine="arm64",
            system="Darwin",
        )
    assert error.value.code == "CODEX_UNSUPPORTED_PLATFORM"
    assert "Install the Codex CLI yourself" in error.value.reason


def test_archive_with_two_members_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("codex", "extra"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    payload = buffer.getvalue()
    monkeypatch.setattr(
        codex_install,
        "CODEX_ASSETS",
        {
            "x86_64": (
                PinnedAsset(
                    archive="a.tar.gz",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    member="codex-x86_64-unknown-linux-musl",
                    destination="codex",
                ),
            )
        },
    )
    with pytest.raises(CodexInstallError) as error:
        install_codex(tmp_path, fetch=lambda _url: payload, machine="x86_64", system="Linux")
    assert error.value.code == "CODEX_ARCHIVE_UNEXPECTED"


def test_download_failure_is_a_refusal_not_a_crash(
    tmp_path: Path, pinned: dict[str, bytes]
) -> None:
    def fetch(_url: str) -> bytes:
        raise TimeoutError("slow")

    with pytest.raises(CodexInstallError) as error:
        install_codex(tmp_path, fetch=fetch, machine="x86_64", system="Linux")
    assert error.value.code == "CODEX_DOWNLOAD_FAILED"
    assert "tap Install again" in error.value.reason


def test_install_route_returns_the_receipt_and_records_the_mutation(server_box, box_home) -> None:
    server, secret = server_box[0], server_box[1]
    status, body = request(server, "POST", "/v1/provider/codex/install", secret=secret)
    assert status == 200
    assert body["code"] == "CODEX_INSTALLED"
    assert body["path"] == "/fixture/bin/codex"
    assert body["paths"]["codex-code-mode-host"] == "/fixture/bin/codex-code-mode-host"
    records = (box_home / "provider" / "records.jsonl").read_text(encoding="utf-8")
    assert "codex_installed" in records
