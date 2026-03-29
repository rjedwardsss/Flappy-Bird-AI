"""
Flappy Bird AI — neuroevolution-style flock (extracted from portfolio logic).

SOURCE (read-only extraction):
    app/projects/flappy/FlappyClient.tsx

This module contains the same numerical / algorithmic behavior as that file:
    - Tiny feedforward neural network (5 → tanh → 6 → sigmoid → 1 flap decision)
    - Genetic operators: random init, copy, per-weight mutation
    - Population simulation: pipes, physics, collision, pipe-pass fitness
    - Selection: sort by (pipes_passed * 1000 + frame_survival_score), elite brains,
      next generation from mutated elites

NOT INCLUDED (by design, not in this extraction scope):
    - Canvas rendering, React, or any UI
    - requestAnimationFrame / real-time display
    - The portfolio Navbar/Footer or page shell

The example at the bottom runs a bounded headless simulation so you can see
generations advance without graphics. For pixel-identical behavior to the
browser build, you would also need identical RNG sequencing (browser vs
Python random differs unless you fix seeds and accept statistical similarity).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- Constants (match FlappyClient.tsx) -------------------------------------

W = 720
H = 420
GRAVITY = 0.4
FLAP_V = -7.5
PIPE_W = 60
GAP_MIN = 105
GAP_MAX = 165
PIPE_SPACING = 160
PIPE_SPEED = 2.6
POP_SIZE = 24
ELITE_COUNT = 6
MUTATION_RATE = 0.15
MUTATION_SCALE = 0.35
INPUTS = 5
HIDDEN = 6
OUTPUTS = 1


# --- Data structures --------------------------------------------------------


@dataclass
class Brain:
    """Small MLP weights: W1 (HIDDEN x INPUTS), b1, W2 (OUTPUTS x HIDDEN), b2."""

    W1: List[List[float]]
    b1: List[float]
    W2: List[List[float]]
    b2: List[float]


@dataclass
class Pipe:
    x: float
    gap_y: float
    gap_h: float
    w: float = PIPE_W
    passed: bool = False


@dataclass
class Bird:
    """One agent: position, velocity, survival counters, and a neural brain."""

    x: float = 110.0
    y: float = H / 2
    vy: float = 0.0
    alive: bool = True
    score: int = 0  # frames survived (incremented each physics step while alive)
    passed: int = 0  # pipes cleared
    brain: Brain = field(default_factory=lambda: Bird.random_brain())

    @staticmethod
    def random_brain() -> Brain:
        return Brain(
            W1=rand_mat(HIDDEN, INPUTS, 0.6),
            b1=rand_vec(HIDDEN, 0.3),
            W2=rand_mat(OUTPUTS, HIDDEN, 0.6),
            b2=rand_vec(OUTPUTS, 0.3),
        )

    @staticmethod
    def copy_brain(brain: Brain) -> Brain:
        return Brain(
            W1=[row[:] for row in brain.W1],
            b1=brain.b1[:],
            W2=[row[:] for row in brain.W2],
            b2=brain.b2[:],
        )

    @staticmethod
    def mutate(brain: Brain) -> Brain:
        return Brain(
            W1=[[mutate_value(v) for v in row] for row in brain.W1],
            b1=[mutate_value(v) for v in brain.b1],
            W2=[[mutate_value(v) for v in row] for row in brain.W2],
            b2=[mutate_value(v) for v in brain.b2],
        )

    @staticmethod
    def forward(brain: Brain, x: List[float]) -> List[float]:
        """
        Forward pass: hidden layer uses tanh; output uses logistic sigmoid.
        Single output in (0, 1): flap if > 0.5 (same threshold as TS).
        """
        h = [0.0] * HIDDEN
        for j in range(HIDDEN):
            s = brain.b1[j]
            for i in range(INPUTS):
                s += brain.W1[j][i] * x[i]
            h[j] = math.tanh(s)

        out = [0.0] * OUTPUTS
        for k in range(OUTPUTS):
            s = brain.b2[k]
            for j in range(HIDDEN):
                s += brain.W2[k][j] * h[j]
            out[k] = 1.0 / (1.0 + math.exp(-s))
        return out

    def think(self, nearest: Optional[Pipe]) -> None:
        """
        Build normalized inputs from state + nearest pipe, then maybe flap.
        Inputs: bird y, vy, gap top/bottom, horizontal distance to pipe.
        """
        px = nearest.x if nearest else W
        gap_y = nearest.gap_y if nearest else H / 2
        gap_h = nearest.gap_h if nearest else 140.0

        x1 = self.y / H
        x2 = self.vy / 10.0
        x3 = (gap_y - gap_h / 2) / H
        x4 = (gap_y + gap_h / 2) / H
        x5 = (px - self.x) / W

        out = Bird.forward(self.brain, [x1, x2, x3, x4, x5])[0]
        if out > 0.5:
            self.flap()

    def flap(self) -> None:
        self.vy = FLAP_V

    def update(self) -> None:
        self.vy += GRAVITY
        self.y += self.vy
        self.score += 1
        if self.y < 8 or self.y > H - 8:
            self.alive = False


# --- Random / mutation helpers (match TS helpers) -----------------------------


def mutate_value(v: float) -> float:
    if random.random() < MUTATION_RATE:
        return v + (random.random() * 2 - 1) * MUTATION_SCALE
    return v


def rand_mat(rows: int, cols: int, s: float) -> List[List[float]]:
    return [[(random.random() * 2 - 1) * s for _ in range(cols)] for _ in range(rows)]


def rand_vec(n: int, s: float) -> List[float]:
    return [(random.random() * 2 - 1) * s for _ in range(n)]


def rand_between(a: float, b: float) -> float:
    return a + random.random() * (b - a)


# --- Simulation core (headless counterpart to TS update / evolution) -------


class FlappySimulation:
    """
    Holds world state: pipes, flock, frame counter, generation, shared score.
    `step` applies `sim_speed` inner physics ticks (same nested loop as TS `update`).
    """

    def __init__(self) -> None:
        self.pipes: List[Pipe] = []
        self.flock: List[Bird] = []
        self.frame = 0
        self.generation = 1
        self.score = 0  # shared pipe-pass counter (matches scoreRef)
        self.best_score_ever = 0

    def spawn_pipe(self) -> None:
        gap_h = rand_between(GAP_MIN, GAP_MAX)
        gap_y = rand_between(100, H - 100)
        self.pipes.append(Pipe(x=W + 20, gap_y=gap_y, gap_h=gap_h, w=PIPE_W, passed=False))

    def reset_world(self) -> None:
        self.pipes = []
        self.frame = 0
        self.score = 0
        self.spawn_pipe()

    def make_flock(self, seed_brains: Optional[List[Brain]] = None) -> None:
        """If seed_brains is None or empty, random brains; else mutate from elites."""
        self.flock = []
        if not seed_brains:
            for _ in range(POP_SIZE):
                self.flock.append(Bird(brain=Bird.random_brain()))
            return

        n = len(seed_brains)
        for i in range(POP_SIZE):
            brain = Bird.mutate(Bird.copy_brain(seed_brains[i % n]))
            self.flock.append(Bird(brain=brain))

    def _fitness(self, b: Bird) -> float:
        return b.passed * 1000 + b.score

    def next_generation(self) -> None:
        """Elite selection + mutated offspring; same ordering as TS."""
        ranked = sorted(self.flock, key=self._fitness, reverse=True)
        elites = [Bird.copy_brain(b.brain) for b in ranked[:ELITE_COUNT]]
        if ranked:
            self.best_score_ever = max(
                self.best_score_ever,
                ranked[0].passed * 1000 + ranked[0].score,
            )
        self.generation += 1
        self.reset_world()
        self.make_flock(elites)

    def _nearest_pipe(self) -> Optional[Pipe]:
        for p in self.pipes:
            if p.x + p.w > 100:
                return p
        return None

    def step(self, sim_speed: int = 1) -> None:
        """
        One call to `step` = `sim_speed` inner iterations (TS `update` inner for-loop).
        Advances physics, AI decisions, collisions; may trigger next_generation.
        """
        for _ in range(sim_speed):
            self.frame += 1

            if self.frame % PIPE_SPACING == 0:
                self.spawn_pipe()

            for p in self.pipes:
                p.x -= PIPE_SPEED
            self.pipes = [p for p in self.pipes if p.x + p.w > -10]

            nearest = self._nearest_pipe()
            alive_count = 0

            for b in self.flock:
                if not b.alive:
                    continue

                b.think(nearest)
                b.update()

                if nearest is not None:
                    if b.x > nearest.x - 10 and b.x < nearest.x + nearest.w + 10:
                        in_gap = nearest.gap_y - nearest.gap_h / 2 < b.y < nearest.gap_y + nearest.gap_h / 2
                        if not in_gap:
                            b.alive = False

                    if not nearest.passed and b.x > nearest.x + nearest.w:
                        nearest.passed = True
                        b.passed += 1
                        self.score += 1

                if b.alive:
                    alive_count += 1

            if alive_count == 0:
                self.next_generation()
                break


# --- Example ----------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    sim = FlappySimulation()
    sim.reset_world()
    sim.make_flock()

    print("Headless Flappy Bird AI demo (neuroevolution + tiny NN, no graphics).")
    print("Constants: population=%d, elites=%d, pipe_speed=%s" % (POP_SIZE, ELITE_COUNT, PIPE_SPEED))
    print()

    max_steps = 120_000
    log_every = 20_000

    for i in range(max_steps):
        sim.step(sim_speed=1)
        if i > 0 and i % log_every == 0:
            alive = sum(1 for b in sim.flock if b.alive)
            print(
                f"  inner_step={i:6d}  gen={sim.generation:3d}  alive={alive:2d}  "
                f"pipe_score={sim.score:4d}  best_ever={sim.best_score_ever}"
            )

    alive = sum(1 for b in sim.flock if b.alive)
    print()
    print(f"After {max_steps} inner steps:")
    print(f"  generation={sim.generation}")
    print(f"  birds_alive={alive}")
    print(f"  current_pipe_pass_score={sim.score}")
    print(f"  best_composite_fitness_ever={sim.best_score_ever}")
