"""
Handmade fishing lure catalog generator (Part 1).

The catalog is intentionally:
- small (100 products by default)
- deterministic (seeded RNG)
- realistic enough to make semantic search "feel" real

We generate product descriptions with the kind of language anglers use:
species, water clarity, cover, action, depth, and target conditions.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    category: str
    species: str
    water: str
    depth: str
    color: str
    weight_oz: float
    price_usd: float
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "species": self.species,
            "water": self.water,
            "depth": self.depth,
            "color": self.color,
            "weight_oz": self.weight_oz,
            "price_usd": self.price_usd,
            "description": self.description,
        }


BRANDS = [
    "Small Batch Baits",
    "Creekside Customs",
    "Hand-Poured Co.",
    "Tin & Tackle",
    "Backwater Lab",
]

CATEGORIES = {
    "soft_plastic": {
        "items": ["worm", "stick bait", "creature bait", "swimbait", "grub"],
        "actions": ["subtle tail kick", "slow fall", "natural glide", "tight wobble"],
        "rigs": ["Texas rig", "wacky rig", "Ned rig", "jig head"],
    },
    "crankbait": {
        "items": ["squarebill", "mid-diver", "lipless crankbait"],
        "actions": ["wide wobble", "tight vibration", "erratic deflection"],
        "rigs": ["straight retrieve", "stop-and-go", "burn and pause"],
    },
    "jerkbait": {
        "items": ["suspending jerkbait", "floating jerkbait"],
        "actions": ["darting action", "long pause suspend", "quick snap"],
        "rigs": ["twitch-twitch-pause", "jerk and pause", "slow cadence"],
    },
    "topwater": {
        "items": ["popper", "walking bait", "buzzbait"],
        "actions": ["walk-the-dog", "spit and pop", "surface buzz"],
        "rigs": ["steady walk", "pop-pause", "burn across flats"],
    },
}

SPECIES = ["bass", "trout", "pike", "walleye", "panfish"]
WATER = ["clear water", "stained water", "muddy water"]
DEPTH = ["topwater", "shallow", "mid-depth", "deep"]
COLORS = ["shad", "bluegill", "crawfish", "black", "white", "chartreuse"]
COVER = ["grass", "rocks", "wood", "weed edges", "open water"]


def generate_products(count: int, *, seed: int = 42) -> list[Product]:
    if count <= 0:
        raise ValueError("count must be > 0")

    rng = random.Random(seed)
    category_keys = list(CATEGORIES.keys())

    products: list[Product] = []
    for pid in range(1, count + 1):
        cat = rng.choice(category_keys)
        spec = CATEGORIES[cat]

        item = rng.choice(spec["items"])
        action = rng.choice(spec["actions"])
        rig = rng.choice(spec["rigs"])

        species = rng.choice(SPECIES)
        water = rng.choice(WATER)
        depth = rng.choice(DEPTH)
        color = rng.choice(COLORS)
        cover = rng.choice(COVER)
        brand = rng.choice(BRANDS)

        weight_oz = round(rng.choice([0.125, 0.1875, 0.25, 0.375, 0.5, 0.625]), 4)
        price_usd = round(rng.uniform(6.99, 18.99), 2)

        name = f"{brand} {color.title()} {item.title()}"
        description = (
            f"Handmade {item} for {species} in {water}. "
            f"Best at {depth} around {cover}. "
            f"Built for {action} with a {rig} presentation. "
            f"Small-batch pour, tuned for consistent action."
        )

        products.append(
            Product(
                id=pid,
                name=name,
                category=cat,
                species=species,
                water=water,
                depth=depth,
                color=color,
                weight_oz=weight_oz,
                price_usd=price_usd,
                description=description,
            )
        )

    return products


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    p = Path(path)
    out: list[dict[str, object]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

