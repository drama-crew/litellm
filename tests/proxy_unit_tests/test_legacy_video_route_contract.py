import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_ROOT = REPO_ROOT / "litellm" / "proxy"
PROXY_SERVER = PROXY_ROOT / "proxy_server.py"
LEGACY_ROUTER = PROXY_ROOT / "libtv_video_endpoints"


def test_legacy_libtv_video_router_is_not_mounted():
    """The retired router must not bypass Causyn admission and staging gates."""
    source = PROXY_SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROXY_SERVER))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert not any("libtv_video_endpoints" in module for module in imported_modules)
    assert "libtv_video_endpoints" not in imported_names
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "include_router"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "libtv_video_router"
        for node in ast.walk(tree)
    )
    assert "/v1/libtv/video-generate" not in source
    assert not any(LEGACY_ROUTER.glob("*.py"))
