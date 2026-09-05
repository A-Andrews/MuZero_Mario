#!/usr/bin/env python
"""Load a MuZero-Mario checkpoint and pull activations out of it.

Standalone: needs only torch + numpy (+ opencv-python if you preprocess your
own raw frames). No stable-retro, no ROM, nothing else from the training repo.

    python load_model.py --list
    python load_model.py --level Level1-1 --npz some_segment.npz --out feats.npz

As a library:

    from load_model import load_model, encode, policy_value
    net = load_model("checkpoints/Level1-1.pt")
    h = encode(net, obs_uint8)              # (B, 192, 6, 6) encoder activations
    probs, value = policy_value(net, obs_uint8)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE / "code"))

from src.muzero.networks import MuZeroNet  # noqa: E402
from src.muzero.transforms import support_to_scalar  # noqa: E402


def load_model(checkpoint: str | Path, device: str = "cpu") -> MuZeroNet:
    """Rebuild the net from the config stored inside the checkpoint itself.

    Do not construct MuZeroNet from defaults: these runs are 192-channel /
    10 dynamics blocks, not the library defaults.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = MuZeroNet(**ck["cfg_snapshot"]["model"])
    net.load_state_dict(ck["online"], strict=True)
    net.to(device).eval()
    net.training_step = ck.get("training_step")
    net.env_step = ck.get("env_step")
    return net


def as_input(obs, device: str = "cpu") -> torch.Tensor:
    """(B, 4, 96, 96) uint8 (or float in [0,1]) -> the float tensor the net expects."""
    t = torch.as_tensor(np.asarray(obs))
    if t.ndim == 3:
        t = t[None]
    t = t.to(device)
    return t.float().div(255.0) if t.dtype == torch.uint8 else t.float()


def frames_to_obs(frames, n_frame: int = 4, downsample: int = 4, pad_to: int = 96):
    """Raw RGB frames -> (N, 4, 96, 96) uint8, one observation per agent step.

    Use this if you are feeding your own stimulus frames rather than the
    supplied .npz corpus. `frames` is a sequence of full-resolution RGB frames
    at the emulator's native 60 Hz, in order, starting from the first frame you
    want a model observation for.

    This reproduces exactly what the training workers and the .bk2 converter do,
    and getting it right by hand is easy to botch:

    - One agent step consumes `downsample` (4) emulator frames, so you get
      len(frames)//4 observations, not len(frames).
    - Each of the 4 channels is the pixel-wise max of two *consecutive* frames
      (NES sprite-flicker removal), taken at strided offsets through a 16-frame
      history — not 4 evenly spaced single frames.
    - The observation for step t is built from the 16 frames *preceding* that
      step's own 4 frames, matching the (s_t, a_t, r_t) convention.
    - The first observation needs 16 frames of history that do not exist, so the
      history is seeded by repeating the opening frame, exactly as the converter
      does on reset and after each respawn. `frames[0]` is that seed frame and
      is NOT itself consumed as step data; steps consume `frames[1:]`, four at a
      time, so you get (len(frames) - 1) // 4 observations.

    Returns uint8 in [0, 255]; pass it straight to `encode` / `policy_value`,
    which divide by 255.
    """
    from collections import deque
    from src.env.preprocess import grayscale_resize, stack_max_pooled

    frames = list(frames)
    if not frames:
        return np.zeros((0, n_frame, pad_to, pad_to), dtype=np.uint8)

    hist = deque(maxlen=n_frame * downsample)
    seed = grayscale_resize(np.asarray(frames[0]))
    for _ in range(n_frame * downsample):
        hist.append(seed)

    rest = frames[1:]
    out = []
    for t in range(len(rest) // downsample):
        # Observation first, then consume this step's frames — (s_t, a_t).
        stacked = stack_max_pooled(hist, n_frame, downsample, pad_to=pad_to)
        out.append((stacked * 255.0).clip(0, 255).astype(np.uint8))
        for f in rest[t * downsample:(t + 1) * downsample]:
            hist.append(grayscale_resize(np.asarray(f)))
    return np.stack(out) if out else np.zeros((0, n_frame, pad_to, pad_to), dtype=np.uint8)


@torch.no_grad()
def encode(net: MuZeroNet, obs, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Encoder activations: (B, hidden_channels, 6, 6), min-max normalised per sample."""
    x = as_input(obs, device)
    return np.concatenate(
        [net.representation(x[i:i + batch_size]).cpu().numpy()
         for i in range(0, len(x), batch_size)], axis=0)


@torch.no_grad()
def policy_value(net: MuZeroNet, obs, device: str = "cpu", batch_size: int = 256):
    """Policy-head probabilities (B, 12) and scalar state value (B,).

    Note this is the *policy head* (the prior), not the MCTS visit distribution
    the agent actually acts on. Reproducing the acted policy needs the search.
    """
    x = as_input(obs, device)
    probs, values = [], []
    for i in range(0, len(x), batch_size):
        h = net.representation(x[i:i + batch_size])
        pl, vl = net.prediction(h)
        probs.append(torch.softmax(pl, dim=-1).cpu().numpy())
        values.append(support_to_scalar(
            vl, net.value_support_min, net.value_support_max, net.value_support_size
        ).cpu().numpy())
    return np.concatenate(probs), np.concatenate(values).reshape(-1)


@torch.no_grad()
def imagine(net: MuZeroNet, obs, action: int, device: str = "cpu") -> np.ndarray:
    """One step of the learned forward model: encoder state -> predicted next state."""
    x = as_input(obs, device)
    h = net.representation(x)
    a = torch.full((len(x),), int(action), dtype=torch.long, device=device)
    h_next, _ = net.dynamics(h, a)
    return h_next.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list bundled checkpoints and exit")
    ap.add_argument("--level", default="Level1-1")
    ap.add_argument("--checkpoint", default=None, help="explicit path, overrides --level")
    ap.add_argument("--npz", default=None,
                    help="an npz with an (N,4,96,96) uint8 'obs_stacks' member")
    ap.add_argument("--frames", default=None,
                    help="a .npy of raw RGB frames (T,H,W,3) at 60 Hz; preprocessed "
                         "for you via frames_to_obs (needs opencv-python)")
    ap.add_argument("--out", default=None, help="write features to this .npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    if args.list:
        print(f"{'level':<10} {'run':<20} {'step':>8} {'sp_rate':>8}  greedy")
        for e in manifest["levels"]:
            g = e.get("greedy_runthrough") or {}
            gs = f"{g.get('n_completed')}/{g.get('n_rollouts')}" if g else "-"
            flag = "" if e.get("completed_level", True) else "  NEVER COMPLETED (latest.pt)"
            print(f"{e['level']:<10} {e['run']:<20} {e['training_step']:>8} "
                  f"{str(e['selfplay_completion_rate']):>8}  {gs}{flag}")
        return 0

    ckpt = args.checkpoint or str(BUNDLE / "checkpoints" / f"{args.level}.pt")
    net = load_model(ckpt, args.device)
    print(f"Loaded {ckpt}  (train step {net.training_step}, env step {net.env_step})")

    if args.frames is not None:
        raw = np.load(args.frames)
        obs = frames_to_obs(raw)
        print(f"Loaded {args.frames}: {raw.shape} raw frames -> {obs.shape} observations")
    elif args.npz is not None:
        obs = np.load(args.npz)["obs_stacks"]
        print(f"Loaded {args.npz}: {obs.shape} {obs.dtype}")
    else:
        obs = np.zeros((2, 4, 96, 96), dtype=np.uint8)
        print("No --npz/--frames given; running a shape check on zeros.")

    h = encode(net, obs, args.device)
    probs, values = policy_value(net, obs, args.device)
    print(f"  encoder  {h.shape}  range [{h.min():.3f}, {h.max():.3f}]")
    print(f"  policy   {probs.shape}   value {values.shape} mean {values.mean():.2f}")

    if args.out:
        np.savez_compressed(args.out, encoder=h, policy=probs, value=values)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
