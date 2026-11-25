"""Tests for sync-dod-skills.py: one Skill + one Agent per DoD item, scope-aware."""

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "sync-dod-skills.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_dod_skills", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_creates_one_skill_per_item_flat_under_skills_root(tmp_path):
    # apm-style whole-repo skill discovery only looks one level under
    # skills/ (skills/<name>/SKILL.md), matching dot-claude's own
    # convention, so generated skills must live directly there too.
    mod = _load_module()
    items = [
        {"id": "a", "title": "A", "check": "check a", "scopes": ["common"]},
        {"id": "b", "title": "B", "check": "check b", "scopes": ["dance"]},
    ]
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    (skills_root / "unrelated-skill").mkdir()  # must survive sync() untouched

    written, removed = mod.sync(items, skills_root)

    assert sorted(written) == ["a", "b"]
    assert removed == []
    dod_dirs = sorted(
        p.name for p in skills_root.iterdir() if p.name.startswith("dod-understanding-")
    )
    assert dod_dirs == ["dod-understanding-a", "dod-understanding-b"]
    assert (skills_root / "unrelated-skill").is_dir()
    content = (skills_root / "dod-understanding-a" / "SKILL.md").read_text(encoding="utf-8")
    assert "name: dod-understanding-a" in content
    assert "check a" in content


def test_sync_renders_scopes_into_generated_content(tmp_path):
    # The whole point of scopes is that a reviewer or the Stop hook can see,
    # from the generated Skill/Agent alone, which directories a check
    # applies to -- it must not require reading the JSON manifest too.
    mod = _load_module()
    items = [{"id": "a", "title": "A", "check": "check a", "scopes": ["dance", "common"]}]
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    agents_root = tmp_path / "agents"
    agents_root.mkdir()

    mod.sync(items, skills_root)
    mod.sync_agents(items, agents_root)

    skill_content = (skills_root / "dod-understanding-a" / "SKILL.md").read_text(encoding="utf-8")
    agent_content = (agents_root / "dod-understanding-a.md").read_text(encoding="utf-8")
    assert "dance" in skill_content and "common" in skill_content
    assert "dance" in agent_content and "common" in agent_content


def test_sync_agents_description_quoting_is_yaml_safe(tmp_path):
    # A literal double quote embedded around the title inside an
    # already-double-quoted YAML scalar (description: "...") breaks
    # frontmatter parsing. Guard against reintroducing that.
    mod = _load_module()
    items = [{"id": "a", "title": "No piece-movement rules", "check": "c", "scopes": ["chess"]}]
    agents_root = tmp_path / "agents"
    agents_root.mkdir()

    mod.sync_agents(items, agents_root)

    content = (agents_root / "dod-understanding-a.md").read_text(encoding="utf-8")
    description_line = next(
        line for line in content.splitlines() if line.startswith("description:")
    )
    # Exactly the two quotes that wrap the whole YAML scalar value.
    assert description_line.count('"') == 2


def test_sync_removes_skills_no_longer_in_items(tmp_path):
    mod = _load_module()
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    (skills_root / "unrelated-skill").mkdir()
    mod.sync(
        [
            {"id": "a", "title": "A", "check": "check a", "scopes": ["common"]},
            {"id": "b", "title": "B", "check": "check b", "scopes": ["common"]},
        ],
        skills_root,
    )

    written, removed = mod.sync(
        [{"id": "a", "title": "A", "check": "check a", "scopes": ["common"]}],
        skills_root,
    )

    assert written == ["a"]
    assert removed == ["b"]
    dod_dirs = sorted(
        p.name for p in skills_root.iterdir() if p.name.startswith("dod-understanding-")
    )
    assert dod_dirs == ["dod-understanding-a"]
    assert (skills_root / "unrelated-skill").is_dir()


def test_sync_agents_creates_one_agent_file_per_item(tmp_path):
    mod = _load_module()
    items = [
        {"id": "a", "title": "A", "check": "check a", "scopes": ["common"]},
        {"id": "b", "title": "B", "check": "check b", "scopes": ["common"]},
    ]
    agents_root = tmp_path / "agents"
    agents_root.mkdir()
    (agents_root / "unrelated-agent.md").write_text("unrelated", encoding="utf-8")

    written, removed = mod.sync_agents(items, agents_root)

    assert sorted(written) == ["a", "b"]
    assert removed == []
    dod_files = sorted(
        p.name for p in agents_root.iterdir() if p.name.startswith("dod-understanding-")
    )
    assert dod_files == ["dod-understanding-a.md", "dod-understanding-b.md"]
    assert (agents_root / "unrelated-agent.md").read_text(encoding="utf-8") == "unrelated"
    content = (agents_root / "dod-understanding-a.md").read_text(encoding="utf-8")
    assert "name: dod-understanding-a" in content
    assert "check a" in content


def test_sync_agents_removes_agent_files_no_longer_in_items(tmp_path):
    mod = _load_module()
    agents_root = tmp_path / "agents"
    agents_root.mkdir()
    (agents_root / "unrelated-agent.md").write_text("unrelated", encoding="utf-8")
    mod.sync_agents(
        [
            {"id": "a", "title": "A", "check": "check a", "scopes": ["common"]},
            {"id": "b", "title": "B", "check": "check b", "scopes": ["common"]},
        ],
        agents_root,
    )

    written, removed = mod.sync_agents(
        [{"id": "a", "title": "A", "check": "check a", "scopes": ["common"]}],
        agents_root,
    )

    assert written == ["a"]
    assert removed == ["b"]
    dod_files = sorted(
        p.name for p in agents_root.iterdir() if p.name.startswith("dod-understanding-")
    )
    assert dod_files == ["dod-understanding-a.md"]
    assert (agents_root / "unrelated-agent.md").is_file()


def test_load_items_reads_json(tmp_path):
    mod = _load_module()
    dod_json = tmp_path / "deliverables-dod.json"
    dod_json.write_text(
        json.dumps({"items": [{"id": "x", "title": "X", "check": "c", "scopes": ["common"]}]}),
        encoding="utf-8",
    )

    items = mod.load_items(dod_json)

    assert items == [{"id": "x", "title": "X", "check": "c", "scopes": ["common"]}]
