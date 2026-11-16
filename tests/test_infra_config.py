"""
Pin how the LiteLLM gateway gets its config (see CLAUDE.md, Decisions → Docker).

config.yaml is baked into a thin image instead of bind-mounted. A bind mount (file or folder)
goes stale when the host path is replaced while the container runs, and the gateway then
silently serves no models. These tests fail if someone brings the mount back, or lets
infra/.env leak into the image's build context.
"""
import re
from pathlib import Path

from rag.config import PROJECT_ROOT

INFRA = Path(PROJECT_ROOT) / "infra"


def _litellm_service() -> str:
    """The `litellm:` block of docker-compose.yml (up to the next top-level service)."""
    text = (INFRA / "docker-compose.yml").read_text(encoding="utf-8")
    match = re.search(r"^  litellm:\n(.*?)(?=^  \w[\w-]*:\n|\Z)", text, flags=re.M | re.S)
    assert match, "litellm service not found in docker-compose.yml"
    return match.group(1)


def _active_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def test_litellm_is_built_from_dockerfile():
    service = "\n".join(_active_lines(_litellm_service()))
    assert "dockerfile: litellm.Dockerfile" in service
    assert "pull_policy: build" in service


def test_config_is_not_bind_mounted():
    for line in _active_lines(_litellm_service()):
        assert "/app/config" not in line or "--config" in line, f"config is mounted again: {line.strip()}"


def test_dockerfile_copies_config_to_the_path_the_command_reads():
    dockerfile = (INFRA / "litellm.Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^COPY\s+config/config\.yaml\s+/app/config/config\.yaml\s*$", dockerfile, flags=re.M)
    assert '"--config", "/app/config/config.yaml"' in _litellm_service()


def test_build_context_excludes_env_files():
    lines = _active_lines((INFRA / ".dockerignore").read_text(encoding="utf-8"))
    assert lines[0] == "*", "build context must start from ignore-everything"
    assert all(line.startswith("!config") for line in lines[1:]), "only config/ may be re-included"
