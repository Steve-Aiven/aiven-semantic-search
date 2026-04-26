"""
Catalog generator for Part 1 of the tutorial series.

This module produces the dataset that gets indexed into OpenSearch. Understanding
what goes into a document matters for semantic search, so it is worth spending
a moment here before moving on to the embedding and indexing code.

Why generate data instead of using a real dataset?
---------------------------------------------------
Using generated data keeps every step of this tutorial reproducible. If you and
a colleague both run `generate-catalog --seed 42`, you get the same 100 products,
the same descriptions, and (after indexing) the same vector space. That makes it
easy to compare notes, screenshots, and search results without diverging.

What is JSONL and why use it?
------------------------------
JSONL (JSON Lines) stores one JSON object per line. It is a good fit here because:
- You can inspect the file with standard tools (`head`, `tail`, `jq`).
- The indexing step can stream rows without loading the whole file into memory,
  which matters when you scale to larger catalogs in later parts of the series.
- It is easy to append new products without re-writing the whole file.

What makes a good document for semantic search?
-----------------------------------------------
The `description` field is what gets embedded into a vector. Everything else
(sku, species, depth, price, etc.) is stored alongside the vector as structured
metadata. In later parts of the series we will use that metadata for filtering
and faceted results. For now the key point is: richer, more natural descriptions
produce better embeddings, which produces better search results.
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
    sku: str         # e.g. "LURE-SO-0001" - deterministic, useful for filtering later
    name: str
    category: str    # broad lure type used as a structured filter field
    species: str     # target fish species
    water: str       # water clarity the lure is designed for
    depth: str       # depth zone
    color: str
    style: str       # specific lure style within the category
    material: str    # what it is made from
    buoyancy: str    # how it behaves at rest (sinking, floating, suspending)
    hook: str        # hook type and size - matters for rigging
    size_in: float   # body length in inches
    weight_oz: float
    batch_size: int  # how many units were made in one pour/production run
    inventory: int   # units currently in stock
    lead_time_days: int  # days to ship if out of stock (0 = in stock / ships today)
    price_usd: float
    description: str  # this is the field that will be embedded into a vector

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "sku": self.sku,
            "name": self.name,
            "category": self.category,
            "species": self.species,
            "water": self.water,
            "depth": self.depth,
            "color": self.color,
            "style": self.style,
            "material": self.material,
            "buoyancy": self.buoyancy,
            "hook": self.hook,
            "size_in": self.size_in,
            "weight_oz": self.weight_oz,
            "batch_size": self.batch_size,
            "inventory": self.inventory,
            "lead_time_days": self.lead_time_days,
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

# CATALOG defines what is realistic for each lure category. Each key maps to
# attributes specific to that type, so the generated descriptions use the right
# terminology (e.g. crankbaits have treble hooks; soft plastics have rigs like
# Texas or wacky). This mirrors how a real product catalog would be structured.
CATALOG: dict[str, dict[str, list[str | float]]] = {
    "soft_plastic": {
        "styles": ["worm", "stick bait", "creature bait", "paddle-tail swimbait", "grub"],
        "materials": ["hand-poured plastisol", "salted plastisol", "floating plastisol"],
        "rigs": ["Texas rig", "wacky rig", "Ned rig", "jig head", "Carolina rig"],
        "actions": ["subtle tail kick", "slow fall", "natural glide", "tight wobble"],
        "buoyancy": ["sinking", "slow-sinking", "floating"],
        "hooks": ["EWG hook", "offset worm hook", "jig hook"],
        "sizes_in": [2.5, 3.0, 3.5, 4.0, 4.8, 5.5],
    },
    "crankbait": {
        "styles": ["squarebill", "lipless crankbait", "mid-diver"],
        "materials": ["resin", "hardwood"],
        "rigs": ["straight retrieve", "stop-and-go", "burn and pause", "deflect off cover"],
        "actions": ["wide wobble", "tight vibration", "erratic deflection"],
        "buoyancy": ["floating", "slow-floating"],
        "hooks": ["#6 trebles", "#4 trebles", "#2 trebles"],
        "sizes_in": [1.5, 2.0, 2.5, 3.0],
    },
    "jerkbait": {
        "styles": ["suspending jerkbait", "floating jerkbait"],
        "materials": ["resin", "hardwood"],
        "rigs": ["twitch-twitch-pause", "jerk and pause", "slow cadence"],
        "actions": ["darting action", "long pause suspend", "quick snap"],
        "buoyancy": ["suspending", "floating"],
        "hooks": ["#6 trebles", "#4 trebles"],
        "sizes_in": [2.5, 3.0, 3.5, 4.0],
    },
    "topwater": {
        "styles": ["popper", "walking bait", "buzzbait"],
        "materials": ["resin", "hardwood", "wire + blade"],
        "rigs": ["steady walk", "pop-pause", "burn across flats", "slow crawl"],
        "actions": ["walk-the-dog", "spit and pop", "surface buzz"],
        "buoyancy": ["floating"],
        "hooks": ["#4 trebles", "#2 trebles", "single hook"],
        "sizes_in": [2.5, 3.0, 3.5, 4.25],
    },
}

SPECIES  = ["bass", "trout", "pike", "walleye", "panfish"]
WATER    = ["clear water", "stained water", "muddy water"]
DEPTH    = ["topwater", "0-3 ft (shallow)", "3-8 ft (mid)", "8-15+ ft (deep)"]
COLORS   = ["shad", "bluegill", "crawfish", "black", "white", "chartreuse",
            "green pumpkin", "watermelon red"]
COVER    = ["grass", "rocks", "wood", "weed edges", "docks", "open water", "laydowns"]
SEASON   = ["spring", "summer", "fall", "winter"]
CLARITY  = ["high visibility", "moderate visibility", "low visibility"]
RETRIEVE = ["slow roll", "steady retrieve", "yo-yo", "deadstick", "rip and pause", "walk it"]


def generate_products(count: int, *, seed: int = 42) -> list[Product]:
    """
    Generate a deterministic set of fishing lure products.

    Seeding the RNG with a fixed value means every run with the same `seed`
    produces the exact same catalog. This is important for the tutorial series:
    search results and scores will be consistent between runs and between readers.

    If you want to experiment with a different catalog, change the seed value.
    The structure and vocabulary stay the same; only the specific combinations
    of attributes change.
    """
    if count <= 0:
        raise ValueError("count must be > 0")

    rng = random.Random(seed)
    category_keys = list(CATALOG.keys())

    products: list[Product] = []
    for pid in range(1, count + 1):
        cat = rng.choice(category_keys)
        spec = CATALOG[cat]

        style    = rng.choice(spec["styles"])
        action   = rng.choice(spec["actions"])
        rig      = rng.choice(spec["rigs"])
        material = rng.choice(spec["materials"])
        buoyancy = rng.choice(spec["buoyancy"])
        hook     = rng.choice(spec["hooks"])
        size_in  = float(rng.choice(spec["sizes_in"]))

        species  = rng.choice(SPECIES)
        water    = rng.choice(WATER)
        depth    = rng.choice(DEPTH)
        color    = rng.choice(COLORS)
        cover    = rng.choice(COVER)
        season   = rng.choice(SEASON)
        clarity  = rng.choice(CLARITY)
        retrieve = rng.choice(RETRIEVE)
        brand    = rng.choice(BRANDS)

        weight_oz      = round(rng.choice([0.125, 0.1875, 0.25, 0.375, 0.5, 0.625]), 4)
        batch_size     = int(rng.choice([6, 8, 10, 12, 16, 20]))
        inventory      = int(rng.randint(0, batch_size))
        lead_time_days = int(rng.choice([0, 1, 2, 3, 5, 7]))

        # Soft plastics are cheaper than hard baits in the real market.
        base      = 7.99 if cat == "soft_plastic" else 11.99
        price_usd = round(base + rng.uniform(0.0, 8.0), 2)

        sku  = f"LURE-{cat[:2].upper()}-{pid:04d}"
        name = f"{brand} {color.title()} {style.title()} ({size_in:.1f}\")"

        # We use three description templates so the catalog reads less like a
        # form and more like natural product copy. The embedding model encodes
        # meaning, not word order, so varied phrasing of the same attributes
        # produces better coverage of the semantic space - exactly what you want
        # when shoppers describe products in unpredictable ways.
        desc_templates = [
            (
                "Small-batch {material} {style} tuned for {species}. "
                "{buoyancy} profile, {hook}, and a {action}. "
                "Best in {water} with {clarity} during {season}. "
                "Fish it {retrieve} at {depth} around {cover} using a {rig}."
            ),
            (
                "Handmade {style} ({size_in:.1f}\") built in {material}. "
                "Designed for {species} when the water is {water}. "
                "{action} and {buoyancy} behavior help it stay in the strike zone. "
                "Target {cover} at {depth}; run a {rig} and use a {retrieve} cadence."
            ),
            (
                "{style_title} made in small batches - {material}, {hook}, and {buoyancy}. "
                "Dialed for {species} in {water} ({clarity}) from {cover}. "
                "Expect a {action}. "
                "Recommended: {rig} with a {retrieve} retrieve, especially in {season}."
            ),
        ]
        description = rng.choice(desc_templates).format(
            material=material,
            style=style,
            style_title=style.title(),
            species=species,
            buoyancy=buoyancy,
            hook=hook,
            action=action,
            water=water,
            clarity=clarity,
            season=season,
            retrieve=retrieve,
            depth=depth,
            cover=cover,
            rig=rig,
            size_in=size_in,
        )

        products.append(
            Product(
                id=pid,
                sku=sku,
                name=name,
                category=cat,
                species=species,
                water=water,
                depth=depth,
                color=color,
                style=style,
                material=material,
                buoyancy=buoyancy,
                hook=hook,
                size_in=size_in,
                weight_oz=weight_oz,
                batch_size=batch_size,
                inventory=inventory,
                lead_time_days=lead_time_days,
                price_usd=price_usd,
                description=description,
            )
        )

    return products


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    """Write rows to a JSONL file, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Read a JSONL file and return all non-empty rows as a list of dicts."""
    p = Path(path)
    out: list[dict[str, object]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
