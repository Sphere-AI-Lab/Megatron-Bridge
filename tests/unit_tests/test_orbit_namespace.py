from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = (
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "scripts",
    REPOSITORY_ROOT / "examples",
    REPOSITORY_ROOT / "tutorials",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
PREVIOUS_CODEBASE_NAME = "sph" + "ere"


def test_orbit_directories_replace_previous_codebase_directories():
    bridge_package = REPOSITORY_ROOT / "src" / "megatron" / "bridge"

    assert (bridge_package / "orbit").is_dir()
    assert not (bridge_package / PREVIOUS_CODEBASE_NAME).exists()
    assert (REPOSITORY_ROOT / "scripts" / "orbit").is_dir()
    assert not (REPOSITORY_ROOT / "scripts" / PREVIOUS_CODEBASE_NAME).exists()


def test_no_previous_codebase_name_remains_in_orbit_source_surface():
    stale_paths = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                if PREVIOUS_CODEBASE_NAME in path.read_text(errors="ignore").casefold():
                    stale_paths.append(path.relative_to(REPOSITORY_ROOT).as_posix())

    assert stale_paths == []
