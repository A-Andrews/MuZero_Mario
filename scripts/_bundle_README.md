# MuZero-Mario — per-level model checkpoints

23 MuZero agents, one per level of NES Super Mario Bros as used in the CNeuroMod
`mario` task. Each was trained from scratch on a single level for 15M
environment steps. They cover **all 22 levels with CNeuroMod human recordings**,
plus Level2-2 (which has a model, but no human ever played it).

`manifest.json` records, per level, which training run produced it, its
training/env step, its self-play completion rate, its greedy run-through score
and a sha256 of every file. Read it before using anything — see
[Three kinds of model](#three-kinds-of-model-in-this-bundle).

Nothing here needs stable-retro, the emulator or the ROM.

---

## Install

```bash
pip install torch numpy      # plus opencv-python if you feed your own frames
python load_model.py --list  # sanity check: lists all 23 models
```

Python 3.9+. CPU is fine; pass `--device cuda` (or `device="cuda"`) if you have
a GPU and are processing a lot of frames.

---

## Getting activations

Two input paths. **Path A** if you have your own stimulus frames; **path B** if
you want our pre-converted human gameplay corpus.

### Path A — you feed your own frames

```python
from load_model import load_model, frames_to_obs, encode, policy_value

net  = load_model("checkpoints/Level1-1.pt")   # or --device "cuda"
obs  = frames_to_obs(my_rgb_frames)            # (T,H,W,3) -> (N,4,96,96) uint8
h    = encode(net, obs)                        # (N, 192, 6, 6)  <- encoder
probs, value = policy_value(net, obs)          # (N, 12), (N,)
```

or from the command line:

```bash
python load_model.py --level Level1-1 --frames my_frames.npy --out feats.npz
```

**What to pass to `frames_to_obs`:**

| | |
|---|---|
| Format | raw **RGB** frames, `(T, H, W, 3)`, uint8 — a list or an ndarray |
| Resolution | native NES `224×240`; any size works, it resizes internally |
| Rate | the emulator's native **60 Hz**, every frame, in order |
| Do **not** | pre-grayscale, pre-resize, normalise, or drop frames — pass them raw |
| You get back | `(N, 4, 96, 96)` uint8 where `N = (T - 1) // 4` |

`frames[0]` is treated as the state at episode start: it seeds the frame
history and is not itself consumed as step data. Every 4 frames after it
produce one observation, because **one agent step = 4 emulator frames**.

**Why not assemble the stack yourself.** The 4 channels are not 4 evenly spaced
frames. Each is the pixel-wise max of two *consecutive* frames (NES sprite
flicker removal), sampled at strided offsets through a 16-frame history; the
observation for step *t* is built from the 16 frames *preceding* that step's own
4; and the first observation seeds non-existent history by repeating the opening
frame. Get any of it wrong and the model still runs and still returns
`(N, 192, 6, 6)` — the activations are just meaningless. `frames_to_obs`
reproduces the training pipeline exactly, verified byte-identical against a
converted human `.bk2` segment (992/992 observations).

**If you must roll your own**, the pipeline is:

```
RGB frame -> grayscale -> resize 84x84 -> /255
          -> max over consecutive frame pairs (flicker removal)
          -> stack 4, at frame_skip 4   (1 agent step = 4 emulator frames)
          -> edge-pad 84 -> 96
```

`code/src/env/preprocess.py` is the exact code the models were trained with.
To check your version against ours, run a few frames through both and compare
with `np.array_equal` — they should match byte for byte, not approximately.

### Path B — the converted human corpus

Available on request (~1.9 GB): the CNeuroMod human `.bk2` recordings already
run through this exact preprocessing. Each `.npz` segment has `obs_stacks`
`(N, 4, 96, 96)` uint8 plus the subject's actual button press per step, with
`sub-XX_ses-XXX_level-wXlX_rep-XXX` in the filename.

```python
import numpy as np
obs = np.load("sub-01_..._seg0.npz")["obs_stacks"]
h   = encode(net, obs)
```

```bash
python load_model.py --level Level1-1 --npz <a_segment>.npz --out feats.npz
```

**Read [fMRI alignment](#fmri-alignment) before building on this** — the corpus
has a known limitation that path A does not.

### What you can extract

| Call | Returns | What it is |
|---|---|---|
| `encode(net, obs)` | `(N, 192, 6, 6)` | encoder output — **the usual target for RSA / encoding models** |
| `policy_value(net, obs)` | `(N, 12)`, `(N,)` | action prior and scalar state value |
| `imagine(net, obs, action)` | `(N, 192, 6, 6)` | learned forward model: predicted next latent |

All are `@torch.no_grad()` and batch internally (`batch_size=256` by default;
lower it if you run out of memory). Inputs may be uint8 or float — `as_input`
divides uint8 by 255, which is what the model expects.

### Four things that bite

1. **Build the net from the checkpoint's own `cfg_snapshot`, never from library
   defaults.** These runs are 192-channel / 10 dynamics blocks; `networks.py`
   defaults to the 256-channel paper net and will not load. `load_model()` does
   this for you — if you construct `MuZeroNet()` yourself, you will get a shape
   error at best and a wrong model at worst.
2. **Observations are uint8 on disk, float/255 at the model boundary.** Feeding
   raw uint8 straight into `net.representation` silently produces garbage.
3. **The encoder output is min-max normalised per sample to [0, 1].** That is a
   MuZero design choice, not a bug, but it means activation *scale* carries no
   information across samples — relevant if your analysis is not scale-invariant.
4. **The policy head is not the agent.** The agent acts on an MCTS visit
   distribution over 50 simulations; `policy_value` returns the prior that seeds
   that search. They differ, sometimes a lot. Reproducing acted behaviour needs
   the search, which this bundle does not include.

---

## What the model is

Three convolutional networks sharing one latent space, 22.6M parameters:

| Component | Signature | What it is |
|---|---|---|
| `net.representation` | (B, 4, 96, 96) → (B, 192, 6, 6) | encoder |
| `net.prediction` | (B, 192, 6, 6) → policy logits (B, 12), value logits | action prior + state value |
| `net.dynamics` | (h, action) → h', reward logits | learned forward model |

Value and reward heads emit categorical support logits, not scalars;
`support_to_scalar` in `code/src/muzero/transforms.py` converts them, and
`policy_value` already does.

---

## Three kinds of model in this bundle

`manifest.json` records the training run behind every level. The difference
matters if you are relating these representations to human brain data:

- **`spec-<level>` — pure self-play (17 of 23).** Never saw human gameplay.
- **`spec-imit-<level>` — human-demonstration rescue (5 of 23: Level1-3,
  Level4-3, Level5-2, Level7-3, Level8-2).** Used where self-play alone never
  completed the level once: a behavioural-cloning pretrain plus an annealed
  human batch mix. **These were trained on the same CNeuroMod subjects'
  gameplay the brain data comes from, so they are confounded** as evidence that
  a model's representations resemble those subjects' brains. Treat them as a
  separate group, or exclude them, depending on the claim you are making.
- **Never completed (1 of 23: Level5-3)**, marked `"checkpoint_kind":
  "latest.pt"` and `"completed_level": false`. Neither recipe ever finished this
  level, so no `best.pt` exists and this is simply the run's final checkpoint.
  Included because a model that plays a level badly is still a model of that
  level — but it is not a competent agent, and it dies at a fixed obstacle
  (x≈907) on every rollout.

---

## Reading the numbers honestly

- **Completion rates are measured with search noise on.** The
  `selfplay_completion_rate` in `manifest.json` comes from MCTS with root
  Dirichlet exploration. Greedy run-throughs are much weaker: **10 of 22
  evaluated levels finish at all**. Level1-1 scores 0.91 in noisy self-play and
  stalls at the same spot from every start; Level8-2 scores 0.52 and finishes
  0 of 5 greedily. `greedy_runthrough` per level in the manifest is the honest
  number.
- These are competent-but-imperfect players, not solved-level agents. For
  representational comparison that is probably fine — just don't describe them
  as having mastered the levels.

---

## fMRI alignment

**Path A (your own frames):** alignment is whatever your stimulus pipeline
already establishes. `frames_to_obs` returns one observation per 4 input
frames, in order, so observation *i* corresponds to input frames
`1+4i … 4+4i` — map those to your timeline as you already do.

**Path B (our corpus): frame-accurate alignment is not currently possible.**
The converter never recorded an absolute `.bk2` frame index. Segments are split
at deaths, and the title-card intro (measured at 124 frames on one w1l1 rep, and
it varies by level and by rep), each death-animation skip, and the short final
window of each segment are all unrecorded. Only per-segment step counts exist,
in `conversion_report.json`. So cumulative agent steps do **not** map onto the
`.bk2` timeline — and because the intro skip is unrecorded too, **even the first
segment is offset**; no observation in the corpus is correctly placed. 75% of
reps carry three such gaps.

Nothing is lost: `.bk2` replay is frame-exact and deterministic, so re-running
the conversion with a frame counter recovers every offset exactly. **If you need
TR-aligned activations from this corpus, ask — it is a small change and a
re-run, not a rebuild.**

---

## Not included

The NES ROM and the `mario.stimuli` integration data are not redistributable —
get them from the CNeuroMod `mario.stimuli` repository directly.
