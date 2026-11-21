"""Materials: constants from `content/materials/*.toml`, each with its source (SPEC 5.1)."""

import re
import tomllib
from pathlib import Path

import pytest

from entropy.core.paths import content_dir
from entropy.world.materials import load_material, parse_material

M1_MATERIALS = ("water", "glass", "concrete")


@pytest.mark.parametrize("name", M1_MATERIALS)
def test_a_material_holds_the_values_of_its_file(name: str) -> None:
    raw = tomllib.loads((content_dir() / "materials" / f"{name}.toml").read_text("utf-8"))

    material = load_material(name)

    assert material.name == raw["name"]
    assert material.density_kg_per_m3 == raw["density_kg_per_m3"]


@pytest.mark.parametrize("name", M1_MATERIALS)
def test_every_constant_cites_a_source(name: str) -> None:
    material = load_material(name)

    assert set(material.sources) == {"density_kg_per_m3"}
    assert all(source.strip() for source in material.sources.values())


def test_water_is_denser_than_air_and_lighter_than_glass() -> None:
    """A sanity check on the files, not a stored constant."""
    water, glass = load_material("water"), load_material("glass")

    assert 1.2 < water.density_kg_per_m3 < glass.density_kg_per_m3


def test_a_constant_without_a_source_is_an_error() -> None:
    text = 'name = "x"\ndensity_kg_per_m3 = 1000.0\n[source]\n'

    with pytest.raises(ValueError, match="density_kg_per_m3 has no source"):
        parse_material(text, origin="x.toml")


def test_unknown_keys_are_errors() -> None:
    text = 'name = "x"\ndensity_kg_per_m3 = 1.0\nyoungs = 3.0\n[source]\ndensity_kg_per_m3 = "s"\n'

    with pytest.raises(ValueError, match=re.escape("x.toml: unknown key youngs")):
        parse_material(text, origin="x.toml")


@pytest.mark.parametrize("density", ["0.0", "-5.0", "nan", '"heavy"'])
def test_an_impossible_density_is_an_error(density: str) -> None:
    text = f'name = "x"\ndensity_kg_per_m3 = {density}\n[source]\ndensity_kg_per_m3 = "s"\n'

    with pytest.raises(ValueError, match="density_kg_per_m3"):
        parse_material(text, origin="x.toml")


def test_a_missing_material_names_its_file() -> None:
    with pytest.raises(
        FileNotFoundError, match=re.escape(str(Path("materials") / "unobtainium.toml"))
    ):
        load_material("unobtainium")
