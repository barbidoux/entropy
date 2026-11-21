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
    raw = tomllib.loads((content_dir() / "materials" / f"{name}.toml").read_text("utf-8"))
    material = load_material(name)

    constants = set(raw) - {"name", "source"}
    assert set(material.sources) == constants
    assert all(source.strip() for source in material.sources.values())


def test_the_test_tank_materials_have_what_the_renderer_needs() -> None:
    """Glass reflects by its refractive index; concrete has a colour (#51)."""
    assert load_material("glass").refractive_index == pytest.approx(1.5)
    albedo = load_material("concrete").albedo_linear_rgb
    assert albedo is not None and all(0.0 < c < 1.0 for c in albedo)


def test_optional_constants_are_read_with_their_sources() -> None:
    text = (
        'name = "x"\ndensity_kg_per_m3 = 1.0\nrefractive_index = 1.33\n'
        "albedo_linear_rgb = [0.1, 0.2, 0.3]\nabsorption_per_m_rgb = [0.3, 0.05, 0.01]\n"
        '[source]\ndensity_kg_per_m3 = "a"\nrefractive_index = "b"\n'
        'albedo_linear_rgb = "c"\nabsorption_per_m_rgb = "d"\n'
    )

    material = parse_material(text, origin="x.toml")

    assert material.refractive_index == 1.33
    assert material.albedo_linear_rgb == (0.1, 0.2, 0.3)
    assert material.absorption_per_m_rgb == (0.3, 0.05, 0.01)
    assert set(material.sources) == {
        "density_kg_per_m3",
        "refractive_index",
        "albedo_linear_rgb",
        "absorption_per_m_rgb",
    }


def test_optional_constants_may_be_left_out() -> None:
    material = parse_material(
        'name = "x"\ndensity_kg_per_m3 = 1.0\n[source]\ndensity_kg_per_m3 = "a"\n',
        origin="x.toml",
    )

    assert material.refractive_index is None
    assert material.albedo_linear_rgb is None
    assert material.absorption_per_m_rgb is None


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("refractive_index = 0.9", "refractive_index"),
        ("refractive_index = nan", "refractive_index"),
        ("albedo_linear_rgb = [0.1, 0.2]", "albedo_linear_rgb"),
        ("albedo_linear_rgb = [0.1, 0.2, 1.5]", "albedo_linear_rgb"),
        ('albedo_linear_rgb = [0.1, 0.2, "x"]', "albedo_linear_rgb"),
        ("absorption_per_m_rgb = [0.1, -0.2, 0.3]", "absorption_per_m_rgb"),
        ("absorption_per_m_rgb = 0.1", "absorption_per_m_rgb"),
    ],
)
def test_an_impossible_optional_constant_is_an_error(line: str, key: str) -> None:
    text = (
        f'name = "x"\ndensity_kg_per_m3 = 1.0\n{line}\n'
        f'[source]\ndensity_kg_per_m3 = "a"\n{key} = "b"\n'
    )

    with pytest.raises(ValueError, match=f"x.toml: {key}"):
        parse_material(text, origin="x.toml")


def test_an_optional_constant_without_a_source_is_an_error() -> None:
    text = (
        'name = "x"\ndensity_kg_per_m3 = 1.0\nrefractive_index = 1.5\n'
        '[source]\ndensity_kg_per_m3 = "a"\n'
    )

    with pytest.raises(ValueError, match="refractive_index has no source"):
        parse_material(text, origin="x.toml")


def test_water_is_lighter_than_glass() -> None:
    """A sanity check on the files, not a stored constant."""
    water, glass = load_material("water"), load_material("glass")

    assert 0.0 < water.density_kg_per_m3 < glass.density_kg_per_m3


def test_a_constant_without_a_source_is_an_error() -> None:
    text = 'name = "x"\ndensity_kg_per_m3 = 1000.0\n[source]\n'

    with pytest.raises(ValueError, match="density_kg_per_m3 has no source"):
        parse_material(text, origin="x.toml")


def test_unknown_keys_are_errors() -> None:
    text = 'name = "x"\ndensity_kg_per_m3 = 1.0\nyoungs = 3.0\n[source]\ndensity_kg_per_m3 = "s"\n'

    with pytest.raises(ValueError, match=re.escape("x.toml: unknown key youngs")):
        parse_material(text, origin="x.toml")


@pytest.mark.parametrize("density", ["0.0", "-5.0", "nan", "inf", "true", '"heavy"'])
def test_an_impossible_density_is_an_error(density: str) -> None:
    text = f'name = "x"\ndensity_kg_per_m3 = {density}\n[source]\ndensity_kg_per_m3 = "s"\n'

    with pytest.raises(ValueError, match="density_kg_per_m3"):
        parse_material(text, origin="x.toml")


def test_a_source_that_is_not_a_table_is_an_error() -> None:
    text = 'name = "x"\ndensity_kg_per_m3 = 1.0\nsource = "somewhere"\n'

    with pytest.raises(ValueError, match=re.escape("x.toml: [source] must be a table")):
        parse_material(text, origin="x.toml")


def test_a_missing_material_names_its_file() -> None:
    with pytest.raises(
        FileNotFoundError, match=re.escape(str(Path("materials") / "unobtainium.toml"))
    ):
        load_material("unobtainium")
