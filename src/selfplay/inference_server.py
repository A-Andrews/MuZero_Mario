"""Central GPU-batched inference server for self-play workers.

Runs as a daemon thread inside the learner process. Owns a shadow copy of the
MuZero network on GPU. Workers send `initial` / `recurrent` inference requests
through a shared mp.Queue and await the reply on a per-worker mp.Pipe. The
server drains up to `max_batch` requests (or waits up to `max_wait_ms`), stacks
them per request-type, runs a single forward pass under `inference_mode`, and
dispatches the split results back.

Learning dynamics are unchanged: the network evaluated here is the same
MuZeroNet class with the same weights that are being trained. The server is a
pure throughput optimisation — it replaces N CPU-resident copies of the network
with one batched GPU inference queue.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.muzero.networks import MuZeroNet


# Request kinds
INITIAL = 0
RECURRENT = 1
SHUTDOWN = 255


def _build_network(cfg_model: Dict[str, Any]) -> MuZeroNet:
    return MuZeroNet(
        input_channels=cfg_model["input_channels"],
        input_spatial=cfg_model["input_spatial"],
        hidden_channels=cfg_model["hidden_channels"],
        hidden_spatial=cfg_model["hidden_spatial"],
        num_actions=cfg_model["num_actions"],
        value_support=tuple(cfg_model["value_support"]),
        reward_support=tuple(cfg_model["reward_support"]),
        rep_blocks=tuple(cfg_model["rep_blocks"]),
        dyn_blocks=cfg_model["dyn_blocks"],
        pred_blocks=cfg_model["pred_blocks"],
    )


class InferenceServer:
    def __init__(
        self,
        cfg_model: Dict[str, Any],
        request_queue,
        reply_conns: List[Any],
        device: torch.device,
        max_batch: int = 256,
        max_wait_ms: float = 1.0,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        wire_dtype: str = "float32",
        pad_batches: bool = False,
    ):
        self.cfg_model = cfg_model
        self.request_queue = request_queue
        self.reply_conns = reply_conns
        self.device = device
        self.max_batch = int(max_batch)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self.use_amp = use_amp and (device.type == "cuda")
        self.amp_dtype = amp_dtype
        # Transport dtype for the (large) hidden-state arrays shipped over the
        # mp.Queue / Pipe. "float32" is byte-identical to the network output;
        # "float16" halves IPC volume at a small precision cost. Scalars
        # (value/reward) always stay float32.
        self._wire_np_dtype = np.dtype(wire_dtype)
        # When True, pad each forward's batch dim up to the next power-of-two
        # bucket (capped at max_batch) so cuDNN autotunes a small fixed set of
        # shapes instead of re-tuning on every batch size. Padding rows are
        # independent of real rows (eval BN uses running stats; hidden-state
        # min-max norm is per-sample) and are sliced off, so replies match the
        # unpadded forward up to float32 conv accumulation-order rounding
        # (~1e-7; the real rows are mathematically unchanged).
        self.pad_batches = bool(pad_batches)

        self._net = _build_network(cfg_model).to(device).eval()
        if self.use_amp:
            self._net = self._net.to(memory_format=torch.channels_last)

        self._weights_lock = threading.Lock()
        self._pending_sd: Optional[Dict[str, torch.Tensor]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API -----------------------------------------------------------

    def update_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Queue a weight update to be applied between batches (thread-safe)."""
        cpu_sd = {k: v.detach().to(self.device, copy=True) for k, v in state_dict.items()}
        with self._weights_lock:
            self._pending_sd = cpu_sd

    def _apply_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Apply weights synchronously. Only safe before the serve thread starts."""
        sd = {k: v.detach().to(self.device, copy=True) for k, v in state_dict.items()}
        self._net.load_state_dict(sd, strict=True)

    def start(self):
        self._thread = threading.Thread(target=self._serve, name="inference-server", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        # Best-effort wake of the queue-get call. The serve loop also exits via
        # the _stop event + the 0.1s get timeout, so a full queue here is fine —
        # never block shutdown on a bounded queue.
        try:
            self.request_queue.put_nowait((-1, SHUTDOWN, None))
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- main loop ------------------------------------------------------------

    def _apply_pending_weights(self):
        with self._weights_lock:
            sd = self._pending_sd
            self._pending_sd = None
        if sd is not None:
            self._net.load_state_dict(sd, strict=True)

    def _serve(self):
        q = self.request_queue
        while not self._stop.is_set():
            self._apply_pending_weights()

            # Block for the first request; then drain opportunistically.
            try:
                first = q.get(timeout=0.1)
            except Exception:
                continue
            if first[1] == SHUTDOWN:
                return

            batch: List[Tuple[int, int, Any]] = [first]
            deadline = time.monotonic() + self.max_wait_s
            while len(batch) < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = q.get(timeout=timeout)
                except Exception:
                    break
                if item[1] == SHUTDOWN:
                    self._stop.set()
                    break
                batch.append(item)

            self._run_batch(batch)

    def _bucket_size(self, n: int) -> int:
        """Smallest power-of-two >= n, capped at max_batch (and >= n)."""
        if not self.pad_batches or n <= 0:
            return n
        b = 1
        while b < n:
            b <<= 1
        return min(max(b, n), self.max_batch)

    def _run_batch(self, batch: List[Tuple[int, int, Any]]):
        init_idx: List[int] = []
        init_obs: List[np.ndarray] = []
        recur_idx: List[int] = []
        recur_h: List[np.ndarray] = []
        recur_a: List[int] = []

        for i, (_wid, kind, payload) in enumerate(batch):
            if kind == INITIAL:
                init_idx.append(i)
                init_obs.append(payload)  # np.ndarray (C,H,W) float32
            elif kind == RECURRENT:
                recur_idx.append(i)
                h_np, a = payload
                recur_h.append(h_np)  # np.ndarray (C,H,W) float32
                recur_a.append(int(a))

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.use_amp
            else _NullCtx()
        )

        replies: List[Optional[Tuple[Any, ...]]] = [None] * len(batch)
        with torch.inference_mode():
            if init_obs:
                n = len(init_obs)
                obs_np = np.stack(init_obs, axis=0)
                bucket = self._bucket_size(n)
                if bucket > n:
                    obs_np = np.concatenate(
                        [obs_np, np.repeat(obs_np[-1:], bucket - n, axis=0)], axis=0
                    )
                obs_t = torch.from_numpy(obs_np).to(self.device, non_blocking=True).float()
                if self.use_amp:
                    obs_t = obs_t.contiguous(memory_format=torch.channels_last)
                with amp_ctx:
                    h, policy_logits, value = self._net.initial_inference(obs_t)
                # Drop padding rows before transport (real rows unchanged).
                h = h[:n]; policy_logits = policy_logits[:n]; value = value[:n]
                h_np = h.detach().float().cpu().numpy().astype(self._wire_np_dtype, copy=False)
                policy_logits_np = policy_logits.float().detach().cpu().numpy()
                value_np = value.float().detach().cpu().numpy()
                for j, idx in enumerate(init_idx):
                    replies[idx] = (
                        h_np[j].copy(),
                        policy_logits_np[j].copy(),
                        float(value_np[j]),
                    )

            if recur_h:
                n = len(recur_h)
                h_np_stack = np.stack(recur_h, axis=0)
                a_list = list(recur_a)
                bucket = self._bucket_size(n)
                if bucket > n:
                    h_np_stack = np.concatenate(
                        [h_np_stack, np.repeat(h_np_stack[-1:], bucket - n, axis=0)], axis=0
                    )
                    a_list = a_list + [a_list[-1]] * (bucket - n)
                h_stack = torch.from_numpy(h_np_stack).to(
                    self.device, non_blocking=True
                ).float()
                a_stack = torch.tensor(a_list, dtype=torch.long, device=self.device)
                if self.use_amp:
                    h_stack = h_stack.contiguous(memory_format=torch.channels_last)
                with amp_ctx:
                    h_next, reward, policy_logits, value = self._net.recurrent_inference(
                        h_stack, a_stack
                    )
                # Drop padding rows before transport (real rows unchanged).
                h_next = h_next[:n]; reward = reward[:n]
                policy_logits = policy_logits[:n]; value = value[:n]
                h_next_np = h_next.detach().float().cpu().numpy().astype(self._wire_np_dtype, copy=False)
                reward_np = reward.float().detach().cpu().numpy()
                policy_logits_np = policy_logits.float().detach().cpu().numpy()
                value_np = value.float().detach().cpu().numpy()
                for j, idx in enumerate(recur_idx):
                    replies[idx] = (
                        h_next_np[j].copy(),
                        float(reward_np[j]),
                        policy_logits_np[j].copy(),
                        float(value_np[j]),
                    )

        for (wid, kind, _payload), reply in zip(batch, replies):
            try:
                self.reply_conns[wid].send(reply)
            except Exception:
                pass


class _NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


# -- Worker-side adapter ------------------------------------------------------


class RemoteNetwork:
    """Stands in for a MuZeroNet inside worker processes.

    Exposes `initial_inference` and `recurrent_inference` with the same policy /
    value return types as the real network (torch tensors, so MCTS can softmax
    / .item() them). The hidden state is kept as a numpy array with a leading
    batch dim: MCTS/Node treat it opaquely (store it, hand it back), so there is
    no reason to round-trip it through torch on every simulation. Hidden states
    cross the queue as numpy arrays anyway, which also avoids
    torch.multiprocessing's shared-memory fd churn on every MCTS step.
    """

    def __init__(self, worker_id: int, request_queue, reply_conn, wire_dtype: str = "float32"):
        self._wid = int(worker_id)
        self._q = request_queue
        self._reply = reply_conn
        self._wire_np_dtype = np.dtype(wire_dtype)

    def eval(self):
        return self

    def initial_inference(self, obs_tensor: torch.Tensor):
        # obs_tensor: (1, C, H, W) float32 on CPU. obs precision is kept fp32
        # (it is cheap — one request per env step, not per simulation).
        obs_np = np.ascontiguousarray(obs_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
        self._q.put((self._wid, INITIAL, obs_np))
        h_np, policy_logits_np, value = self._recv_reply()
        # Hidden state stays numpy; add batch dim back for shape parity.
        h = h_np[None, ...]
        policy_logits_t = torch.from_numpy(policy_logits_np).unsqueeze(0)
        value_t = torch.tensor([value], dtype=torch.float32)
        return h, policy_logits_t, value_t

    def recurrent_inference(self, h_state, action_tensor: torch.Tensor):
        # h_state is the numpy (1, C, H, W) array we returned previously — no
        # torch<->numpy ping-pong. Strip batch dim and ship at the wire dtype.
        h_np = np.ascontiguousarray(h_state[0], dtype=self._wire_np_dtype)
        action_idx = int(action_tensor.item())
        self._q.put((self._wid, RECURRENT, (h_np, action_idx)))
        h_next_np, reward, policy_logits_np, value = self._recv_reply()
        h_next = h_next_np[None, ...]
        reward_t = torch.tensor([reward], dtype=torch.float32)
        policy_logits_t = torch.from_numpy(policy_logits_np).unsqueeze(0)
        value_t = torch.tensor([value], dtype=torch.float32)
        return h_next, reward_t, policy_logits_t, value_t

    def _recv_reply(self):
        try:
            reply = self._reply.recv()
        except (EOFError, OSError) as e:
            raise RuntimeError("inference server shut down") from e
        if reply is None:
            raise RuntimeError("inference server shut down")
        return reply
