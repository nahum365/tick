"""Installing the pinned Codex CLI on the box: verified bytes in, refusals out."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from tests.serve.conftest import request
from tick.serve import codex_install
from tick.serve.codex_install import CodexInstallError, install_codex


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
def pinned(monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload = _tarball()
    monkeypatch.setattr(
        codex_install,
        "CODEX_ASSETS",
        {"x86_64": ("codex-x86_64-unknown-linux-musl.tar.gz", hashlib.sha256(payload).hexdigest())},
    )
    return payload


def test_install_places_an_executable_under_home_bin(tmp_path: Path, pinned: bytes) -> None:
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return pinned

    result = install_codex(tmp_path, fetch=fetch, machine="x86_64", system="Linux")

    binary = tmp_path / "bin" / "codex"
    assert result["code"] == "CODEX_INSTALLED" and result["path"] == str(binary)
    assert binary.read_bytes().startswith(b"#!/bin/sh")
    assert binary.stat().st_mode & 0o111 == 0o111
    assert urls == [
        f"https://github.com/openai/codex/releases/download/{codex_install.CODEX_RELEASE_TAG}"
        "/codex-x86_64-unknown-linux-musl.tar.gz"
    ]
    assert result["reason"].endswith("You can start device login.")


def test_checksum_mismatch_installs_nothing(tmp_path: Path, pinned: bytes) -> None:
    with pytest.raises(CodexInstallError) as error:
        install_codex(tmp_path, fetch=lambda _url: pinned + b"x", machine="x86_64", system="Linux")
    assert error.value.code == "CODEX_CHECKSUM_MISMATCH"
    assert not (tmp_path / "bin").exists()


def test_unsupported_platform_refuses_with_the_next_step(tmp_path: Path, pinned: bytes) -> None:
    with pytest.raises(CodexInstallError) as error:
        install_codex(tmp_path, fetch=lambda _url: pinned, machine="arm64", system="Darwin")
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
        codex_install, "CODEX_ASSETS", {"x86_64": ("a.tar.gz", hashlib.sha256(payload).hexdigest())}
    )
    with pytest.raises(CodexInstallError) as error:
        install_codex(tmp_path, fetch=lambda _url: payload, machine="x86_64", system="Linux")
    assert error.value.code == "CODEX_ARCHIVE_UNEXPECTED"


def test_download_failure_is_a_refusal_not_a_crash(tmp_path: Path, pinned: bytes) -> None:
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
    records = (box_home / "provider" / "records.jsonl").read_text(encoding="utf-8")
    assert "codex_installed" in records
