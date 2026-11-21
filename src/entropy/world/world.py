"""The world built from a map: environment, and solid shapes with their materials.

This is what the simulation reads to set up solvers (SPEC 8: world -> sim). The
solids are static in M1; moving rigid bodies come in M2.
"""

from __future__ import annotations

from dataclasses import dataclass

from entropy.world.maps import Environment, MapData
from entropy.world.materials import Material, load_material
from entropy.world.prefabs import build
from entropy.world.sdf import Shape, Solids


@dataclass(frozen=True)
class World:
    name: str
    environment: Environment
    solids: tuple[tuple[Shape, Material], ...]

    def solid_union(self) -> Solids:
        """Every static solid as one union, for distance queries."""
        return Solids(tuple(shape for shape, _ in self.solids))


def build_world(data: MapData) -> World:
    """The world of `data`: each entity's prefab shapes, made of its material."""
    solids: list[tuple[Shape, Material]] = []
    for index, entity in enumerate(data.entities):
        where = f"entity {index}"
        try:
            material = load_material(entity.material)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"{where}: material {entity.material}: {error}") from error
        solids.extend((shape, material) for shape in build(entity, where))
    return World(name=data.name, environment=data.environment, solids=tuple(solids))
