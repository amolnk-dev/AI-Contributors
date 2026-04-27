"""Astar Island DTOs and constants — the single source of truth for terrain mapping."""

from pydantic import BaseModel
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Terrain codes → prediction class index
# Internal simulation codes map to 6 prediction classes.
# Ocean (10), Plains (11), and Empty (0) all collapse to class 0.
# ──────────────────────────────────────────────────────────────

TERRAIN_TO_CLASS: dict[int, int] = {
    0: 0,    # Empty → class 0
    10: 0,   # Ocean → class 0
    11: 0,   # Plains → class 0
    1: 1,    # Settlement
    2: 2,    # Port
    3: 3,    # Ruin
    4: 4,    # Forest
    5: 5,    # Mountain
}

NUM_CLASSES = 6
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Terrain codes that never change during simulation
STATIC_TERRAIN_CODES = {0, 10, 11, 5}  # empty/ocean/plains/mountain

# Terrain codes where the cell might change
DYNAMIC_TERRAIN_CODES = {1, 2, 3, 4}  # settlement/port/ruin/forest

# Minimum probability floor — prevents KL divergence blowup
# Tuned: 0.01 → 0.001 (+2.5 pts) → 0.0001 (+0.2 pts)
PROB_FLOOR = 0.0001


# ──────────────────────────────────────────────────────────────
# API response models
# ──────────────────────────────────────────────────────────────

class Settlement(BaseModel):
    x: int
    y: int
    population: Optional[float] = None
    food: Optional[float] = None
    wealth: Optional[float] = None
    defense: Optional[float] = None
    has_port: bool = False
    alive: bool = True
    owner_id: Optional[int] = None


class InitialSettlement(BaseModel):
    x: int
    y: int
    has_port: bool = False
    alive: bool = True


class Viewport(BaseModel):
    x: int
    y: int
    w: int
    h: int


class SimResult(BaseModel):
    grid: list[list[int]]
    settlements: list[Settlement]
    viewport: Viewport
    width: int
    height: int
    queries_used: int
    queries_max: int


class InitialState(BaseModel):
    grid: list[list[int]]
    settlements: list[InitialSettlement]


class RoundInfo(BaseModel):
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int = 5
    initial_states: list[InitialState] = []


class RoundSummary(BaseModel):
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int


class BudgetInfo(BaseModel):
    round_id: str
    queries_used: int
    queries_max: int
    active: bool


class AnalysisResult(BaseModel):
    prediction: list[list[list[float]]]
    ground_truth: list[list[list[float]]]
    score: Optional[float]
    width: int
    height: int
    initial_grid: Optional[list[list[int]]]
