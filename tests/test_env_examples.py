"""
Keep the .env.example files in sync with the code, so a fresh clone never misses a setting.

- Every env var the Python code reads must appear in the root .env.example.
- Every var docker-compose.yml / config/config.yaml reads must appear in infra/.env.example.
- Every key in an example must be read somewhere (no stale keys).
Commented-out keys (`# KEY=`) count as documented.
"""
import re
from pathlib import Path

from rag.config import PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
INFRA = ROOT / "infra"
SKIP_DIRS = {".venv", ".git", "tests", "vector_store"}


def _example_keys(path: Path) -> set[str]:
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", path.read_text(encoding="utf-8"), flags=re.M))


def _python_env_reads() -> set[str]:
    found: set[str] = set()
    for path in ROOT.rglob("*.py"):
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        found |= set(re.findall(r'getenv\(\s*"([A-Z][A-Z0-9_]+)"', path.read_text(encoding="utf-8")))
    return found


def _infra_env_reads() -> set[str]:
    found: set[str] = set()
    for path in [INFRA / "docker-compose.yml", INFRA / "config" / "config.yaml"]:
        text = path.read_text(encoding="utf-8")
        found |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]+)", text))
        found |= set(re.findall(r"os\.environ/([A-Z][A-Z0-9_]+)", text))
    return found


def test_root_example_covers_python_settings():
    assert _python_env_reads() - _example_keys(ROOT / ".env.example") == set()


def test_infra_example_covers_compose_and_config():
    assert _infra_env_reads() - _example_keys(INFRA / ".env.example") == set()


def test_examples_have_no_stale_keys():
    assert _example_keys(ROOT / ".env.example") - _python_env_reads() == set()
    assert _example_keys(INFRA / ".env.example") - _infra_env_reads() == set()
