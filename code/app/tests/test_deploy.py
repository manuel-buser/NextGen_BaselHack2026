"""The parts of tools/deploy.py that need no network, no processes and no terminal."""

import hashlib
import io
import tarfile
import zipfile

import pytest

from tools import deploy

TUNNEL_LOG = """2026-10-31T22:01:02Z INF Requesting new quick Tunnel on trycloudflare.com...
2026-10-31T22:01:04Z INF +--------------------------------------------------------------------------------------------+
2026-10-31T22:01:04Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-10-31T22:01:04Z INF |  https://fall-belongs-badly-notre.trycloudflare.com                                       |
2026-10-31T22:01:04Z INF +--------------------------------------------------------------------------------------------+
2026-10-31T22:01:06Z INF Registered tunnel connection connIndex=0 location=zrh01 protocol=quic
"""


@pytest.mark.parametrize(
    ("system", "machine", "uv_file", "cf_file"),
    [
        ("Linux", "x86_64", "uv-x86_64-unknown-linux-gnu.tar.gz", "cloudflared-linux-amd64"),
        ("Linux", "aarch64", "uv-aarch64-unknown-linux-gnu.tar.gz", "cloudflared-linux-arm64"),
        ("Windows", "AMD64", "uv-x86_64-pc-windows-msvc.zip", "cloudflared-windows-amd64.exe"),
        ("Darwin", "x86_64", "uv-x86_64-apple-darwin.tar.gz", "cloudflared-darwin-amd64.tgz"),
        ("Darwin", "arm64", "uv-aarch64-apple-darwin.tar.gz", "cloudflared-darwin-arm64.tgz"),
    ],
)
def test_platform_picks_its_release_files(system, machine, uv_file, cf_file):
    assert deploy.pick(deploy.UV_FILES, "uv", system, machine)[0] == uv_file
    assert deploy.pick(deploy.CF_FILES, "cloudflared", system, machine)[0] == cf_file


def test_unknown_platform_stops_with_a_message():
    with pytest.raises(SystemExit, match="No uv download for Windows aarch64"):
        deploy.pick(deploy.UV_FILES, "uv", "Windows", "ARM64")


def test_checksum_accepts_good_and_rejects_bad_bytes():
    digest = hashlib.sha256(b"program").hexdigest()
    assert deploy.checksum_ok(b"program", digest)
    assert not deploy.checksum_ok(b"program!", digest)


def test_unpack_finds_the_program_in_both_archive_kinds():
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        archive.writestr("uvx.exe", b"other")
        archive.writestr("uv.exe", b"uv program")
    assert deploy.unpack("uv.zip", packed.getvalue(), "uv.exe") == b"uv program"

    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w:gz") as archive:
        for name, data in (("uv-x86_64-apple-darwin/uvx", b"other"), ("uv-x86_64-apple-darwin/uv", b"uv program")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    assert deploy.unpack("uv.tar.gz", packed.getvalue(), "uv") == b"uv program"
    assert deploy.unpack("cloudflared-linux-amd64", b"raw program", "cloudflared") == b"raw program"
    with pytest.raises(SystemExit):
        deploy.unpack("uv.tar.gz", packed.getvalue(), "cloudflared")


def test_link_comes_from_the_created_tunnel_message():
    assert deploy.tunnel_link(TUNNEL_LOG) == "https://fall-belongs-badly-notre.trycloudflare.com"
    assert deploy.tunnel_link('ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel"') is None


@pytest.mark.parametrize(("answer", "expected"), [("", True), ("y", True), (" YES\n", True), ("n", False), ("no", False), ("x", False)])
def test_enter_means_yes(answer, expected):
    assert deploy.is_yes(answer) is expected
