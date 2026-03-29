# Flappy Bird AI

## Overview

This project is a **dependency-free Python** port of the Flappy Bird **neuroevolution** demo from a larger portfolio: a **flock of agents** each controlled by a **small neural network** learns to fly through gaps by **selection, mutation, and elitism** across generations. There is **no graphics** in this repo version—the same physics, collisions, scoring, and evolution run **headlessly** so the AI logic is easy to read and run from the command line.

## How It Works

- **Neural network** — Each bird has a tiny **feedforward network**: **5** normalized inputs (vertical position, velocity, gap bounds, distance to pipe), **6** hidden units with **tanh**, and **1** output with a **logistic sigmoid**. If the output is **greater than 0.5**, the bird **flaps** (same rule as the original implementation).
- **Genetic algorithm** — The population is fixed size (`POP_SIZE`). When **all** birds die, the next generation is built from the **top `ELITE_COUNT` brains** (by fitness). Each new bird gets a **mutated copy** of an elite (random Gaussian-style perturbations on weights with fixed mutation rate and scale).
- **Fitness** — Birds are ranked by **`pipes_passed * 1000 + frames_alive`** (survival score ticks up each step while alive). Higher is better; this matches the sort key used in the source demo.
- **World** — Pipes spawn on a fixed frame interval, move left at constant speed, and kill birds outside the gap when horizontally overlapping. Passing a pipe increments that bird’s pipe count and a global pass counter.

## Features

- **Population-based neuroevolution** with elitism and weight mutation  
- **Alpha-free** small MLP forward pass (tanh + sigmoid)  
- **Headless simulation** class (`FlappySimulation`) mirroring the original update loop  
- **No third-party dependencies** (standard library only)  

## Usage

```bash
python flappy_bird_ai.py
```

To experiment in code, import `FlappySimulation`, `Bird`, `Brain`, or constants from `flappy_bird_ai` and call `sim.reset_world()`, `sim.make_flock()`, and `sim.step(sim_speed=1)` in your own loop.

## Example

The built-in script seeds RNG for reproducibility, creates a simulation, runs a **fixed number of inner physics steps** (`step` with `sim_speed=1`), and prints progress every 20,000 steps: current **generation**, **alive** count, **pipe pass score**, and **best composite fitness** seen so far. This demonstrates evolution advancing without a canvas.

## Why I Built This

It is a compact reference for **embodied decision-making under noise**: cheap **function approximators** (neural nets), **parallel search** over policies (populations), and **iterative improvement** (evolution)—ideas that connect to RL, neuroevolution, and interactive systems more broadly.
