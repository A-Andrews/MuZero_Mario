# MuZero-Mario — per-level model checkpoints

Twelve MuZero agents, one per level of NES Super Mario Bros as used in the
CNeuroMod `mario` task. Each was trained from scratch on a single level for
15M environment steps. `manifest.json` records, per level, which training run
produced it, its training/env step, its self-play completion rate and its
greedy run-through score, plus a sha256 of every file.

## Quick start

```bash
pip install torch numpy          # opencv-python too, only if you preprocess raw frames
python load_model.py --list
python load_model.py --level Level1-1 --npz <a_segment>.npz --out feats.npz
```

```python
from load_model import load_model, encode, policy_value
net = load_model("checkpoints/Level3-3.pt")
h = encode(net, obs)                  # (N, 192, 6, 6) encoder activations
probs, value = policy_value(net, obs) # (N, 12) action prior, (N,) state value
```

Nothing here needs stable-retro, the emulator or the ROM.

## What the model is

Three convolutional networks sharing one latent space:

| Component | Signature | What it is |
|---|---|---|
| `net.representation` | (B, 4, 96, 96) → (B, 192, 6, 6) | encoder; **the usual target for RSA / encoding models** |
| `net.prediction` | (B, 192, 6, 6) → policy logits (B, 12), value logits | action prior + state value |
| `net.dynamics` | (h, action) → h', reward logits | learned forward model, one step of imagination |

22.6M parameters. The encoder output is min-max normalised per sample to
[0, 1] — a MuZero design choice, not a bug, but it means activation *scale*
carries no information across samples.

**Build the net from the checkpoint's own `cfg_snapshot`, never from library
defaults.** These runs are 192-channel / 10 dynamics blocks; the defaults in
`networks.py` are the 256-channel paper net and will not load. `load_model()`
handles this.

## Observations

Input is a 4-frame stack of 96×96 grayscale, stored as `uint8` and divided by
255 at the model boundary (`load_model.as_input` does this). Full pipeline:

    RGB frame -> grayscale -> resize 84x84 -> /255
             -> max over consecutive frame pairs (NES flicker removal)
             -> stack 4, at frame_skip 4  (1 agent step = 4 emulator frames)
             -> edge-pad 84 -> 96

`code/src/env/preprocess.py` is the exact code. If you feed the model frames
preprocessed any other way, the activations are not comparable to anything
these agents were trained on.

## Reading the numbers honestly

- **Completion rates are with search noise on.** The self-play rates in
  `manifest.json` come from MCTS with root Dirichlet exploration. Greedy
  run-throughs are much weaker: 5 of 12 levels finish at all. Level1-1 scores
  0.91 in noisy self-play and stalls at the same spot from every start.
- **The policy head is not the agent.** The agent acts on an MCTS visit
  distribution over 50 simulations; `policy_value` gives the prior that seeds
  that search. They differ, sometimes a lot. Reproducing acted behaviour needs
  the search, which this bundle does not include.
- These are competent-but-imperfect players, not solved-level agents. For
  representational comparison that is probably fine; just don't describe them
  as having mastered the levels.

## If you are aligning to human gameplay

The converted CNeuroMod human corpus (`outputs/human_trajectories/`, shipped
separately on request) has the human .bk2 recordings run through this exact
preprocessing: `obs_stacks` (N, 4, 96, 96) uint8 plus the subject's actual
button press per step, with `sub-XX_ses-XXX_level-wXlX_rep-XXX` in the
filename. That gives model activations time-locked to the frames a subject saw.

**Caveat, important for fMRI alignment:** segments are split at deaths and
title-card / dying / respawn frames are dropped, and no absolute .bk2 frame
index is stored — only per-segment step counts in `conversion_report.json`.
Cumulative agent steps therefore do **not** map linearly back onto the .bk2
timeline. Frame-accurate TR alignment needs a converter change upstream (emit
a frame-index array) and a re-run; ask before building around the current
files.

## Not included

The NES ROM and the `mario.stimuli` integration data are not redistributable —
get them from the CNeuroMod `mario.stimuli` repository directly.
