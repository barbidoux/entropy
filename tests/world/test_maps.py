"""Map files, format v1, and the world built from them (SPEC 4.3, 4.4)."""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from entropy.core.paths import content_dir
from entropy.world.maps import SCHEMA_VERSION, load_map, save_map
from entropy.world.sdf import OpenBox, signed_distance
from entropy.world.world import build_world

TEST_TANK = content_dir() / "maps" / "test_tank"


def _copy_map(tmp_path: Path) -> Path:
    folder = tmp_path / "map"
    shutil.copytree(TEST_TANK, folder)
    return folder


def test_the_test_tank_map_loads() -> None:
    data = load_map(TEST_TANK)

    assert data.name == "Test Tank"
    assert data.schema_version == SCHEMA_VERSION == 1
    assert data.environment.gravity_m_per_s2 == (0.0, 0.0, -9.81)
    assert data.environment.air_temperature_k == pytest.approx(293.15)
    assert sorted(e.prefab for e in data.entities) == ["floor_slab", "glass_tank"]


def test_the_test_tank_is_a_4_by_4_by_2_m_glass_tank_on_concrete() -> None:
    """SPEC 4.4."""
    world = build_world(load_map(TEST_TANK))

    tanks = [(s, m) for s, m in world.solids if isinstance(s, OpenBox)]
    assert len(tanks) == 1
    tank, glass = tanks[0]
    size = np.subtract(tank.inner_max_m, tank.inner_min_m)
    np.testing.assert_allclose(size, (4.0, 4.0, 2.0))
    assert glass.name == "Annealed glass"
    assert {m.name for _, m in world.solids} == {"Annealed glass", "Concrete"}


def test_the_world_answers_distance_queries(device: str) -> None:
    world = build_world(load_map(TEST_TANK))
    tank = next(s for s, _ in world.solids if isinstance(s, OpenBox))
    centre = (np.asarray(tank.inner_min_m) + np.asarray(tank.inner_max_m)) / 2
    points = wp.array([centre.astype(np.float32)], dtype=wp.vec3, device=device)
    distance = wp.zeros(1, dtype=float, device=device)
    normal = wp.zeros(1, dtype=wp.vec3, device=device)

    signed_distance(world.solid_union().to_device(device), points, distance, normal)

    assert distance.numpy()[0] == pytest.approx(1.0, abs=1e-5)  # the cavity's floor is 1 m below


def test_saving_and_loading_gives_the_same_map(tmp_path: Path) -> None:
    data = load_map(TEST_TANK)

    save_map(data, tmp_path / "copy")

    assert load_map(tmp_path / "copy") == data


def test_an_older_schema_is_migrated(tmp_path: Path) -> None:
    """SPEC 4.3: old maps load forever. The registry is tested with a fake v0."""
    folder = _copy_map(tmp_path)
    meta = (
        (folder / "map.toml")
        .read_text(encoding="utf-8")
        .replace("schema_version = 1", "schema_version = 0")
    )
    (folder / "map.toml").write_text(meta, encoding="utf-8")
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    for entity in scene["entities"]:
        entity["where"] = entity.pop("position_m")  # what v0 called it
    (folder / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    def v0_to_v1(raw: dict) -> dict:
        for entity in raw["scene"]["entities"]:
            entity["position_m"] = entity.pop("where")
        return raw

    data = load_map(folder, migrations={0: v0_to_v1})

    assert data.schema_version == 1
    assert data == load_map(TEST_TANK)


def test_a_missing_migration_is_an_error(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    text = (
        (folder / "map.toml")
        .read_text(encoding="utf-8")
        .replace("schema_version = 1", "schema_version = 0")
    )
    (folder / "map.toml").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="no migration from schema 0"):
        load_map(folder, migrations={})


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    text = (
        (folder / "map.toml")
        .read_text(encoding="utf-8")
        .replace("schema_version = 1", "schema_version = 9")
    )
    (folder / "map.toml").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="schema 9 is newer"):
        load_map(folder)


def test_an_unknown_prefab_names_the_entity(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    scene["entities"][0]["prefab"] = "spaceship"
    (folder / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="entity 0: unknown prefab spaceship"):
        build_world(load_map(folder))


def test_an_unknown_material_names_the_entity(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    scene["entities"][1]["material"] = "unobtainium"
    (folder / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="entity 1: material unobtainium"):
        build_world(load_map(folder))


def test_broken_files_name_themselves(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    (folder / "scene.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"scene\.json"):
        load_map(folder)


def test_a_missing_map_folder_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nowhere"):
        load_map(tmp_path / "nowhere")


def test_a_bad_prefab_parameter_names_the_entity(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    scene["entities"][0]["parameters"]["wall_m"] = -0.02
    (folder / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="entity 0: parameter wall_m"):
        build_world(load_map(folder))


def test_the_tank_and_the_slab_sit_where_the_map_says() -> None:
    """Cavity base centred on (0, 0, 0); glass floor 2 cm thick resting on the slab."""
    world = build_world(load_map(TEST_TANK))
    tank = next(s for s, _ in world.solids if isinstance(s, OpenBox))
    slab = next(s for s, _ in world.solids if not isinstance(s, OpenBox))

    np.testing.assert_allclose(tank.inner_min_m, (-2.0, -2.0, 0.0))
    np.testing.assert_allclose(tank.inner_max_m, (2.0, 2.0, 2.0))
    slab_top = slab.centre_m[2] + slab.half_size_m[2]  # type: ignore[union-attr]
    assert slab_top == pytest.approx(tank.inner_min_m[2] - tank.wall_m)


def test_the_glass_floor_and_the_slab_are_solid(device: str) -> None:
    world = build_world(load_map(TEST_TANK))
    points = wp.array(
        np.array([[0, 0, -0.01], [0, 0, -0.1]], np.float32), dtype=wp.vec3, device=device
    )
    distance = wp.zeros(2, dtype=float, device=device)
    normal = wp.zeros(2, dtype=wp.vec3, device=device)

    signed_distance(world.solid_union().to_device(device), points, distance, normal)

    assert np.all(distance.numpy() < 0.0)


def test_migrations_run_one_step_at_a_time_in_order(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    text = (
        (folder / "map.toml")
        .read_text(encoding="utf-8")
        .replace("schema_version = 1", "schema_version = 0")
    )
    (folder / "map.toml").write_text(text, encoding="utf-8")
    steps: list[int] = []

    def step(n: int):
        def run(raw: dict) -> dict:
            assert raw["map"]["schema_version"] == n  # written back after each step
            steps.append(n)
            return raw

        return run

    data = load_map(folder, migrations={0: step(0), 1: step(1)}, current_version=2)

    assert steps == [0, 1]
    assert data.schema_version == 2


@pytest.mark.parametrize("version", ['"1"', "true", "-1"])
def test_an_impossible_schema_version_is_refused(tmp_path: Path, version: str) -> None:
    folder = _copy_map(tmp_path)
    text = (
        (folder / "map.toml")
        .read_text(encoding="utf-8")
        .replace("schema_version = 1", f"schema_version = {version}")
    )
    (folder / "map.toml").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_map(folder)


@pytest.mark.parametrize(
    "name", ["tab\there", "del\x7fhere", 'quote " and \\ backslash', "naïve 🌊"]
)
def test_any_saved_name_loads_back(tmp_path: Path, name: str) -> None:
    data = dataclasses.replace(load_map(TEST_TANK), name=name, description=name)

    save_map(data, tmp_path / "copy")

    assert load_map(tmp_path / "copy") == data


@pytest.mark.parametrize("name", ["   ", "\ud800"])
def test_save_refuses_what_load_would_refuse(tmp_path: Path, name: str) -> None:
    data = dataclasses.replace(load_map(TEST_TANK), name=name)

    with pytest.raises(ValueError, match="name"):
        save_map(data, tmp_path / "copy")


def test_a_description_must_be_text(tmp_path: Path) -> None:
    folder = _copy_map(tmp_path)
    text = (folder / "map.toml").read_text(encoding="utf-8")
    text = text.replace('description = "', 'description = 42\nold_description = "')
    (folder / "map.toml").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="description"):
        load_map(folder)
