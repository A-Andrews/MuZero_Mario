"""Levels eligible for human/model analysis, derived from converted gameplay."""
from pathlib import Path
import re


def human_levels(data_dir="outputs/human_trajectories"):
    levels = set()
    for path in Path(data_dir).glob("sub-*_level-w*l*_*.npz"):
        match = re.search(r"_level-w(\d+)l(\d+)_", path.name)
        if match:
            levels.add(f"Level{match[1]}-{match[2]}")
    if not levels:
        raise ValueError(f"No converted human gameplay found in {data_dir}")
    return sorted(levels, key=lambda level: tuple(map(int, level[5:].split("-"))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", default="outputs/human_trajectories")
    parser.add_argument("--require-level")
    args = parser.parse_args()
    levels = human_levels(args.human_dir)
    if args.require_level:
        if args.require_level not in levels:
            parser.error(f"{args.require_level} has no converted human gameplay and is outside the active project scope")
    else:
        print("\n".join(levels))
