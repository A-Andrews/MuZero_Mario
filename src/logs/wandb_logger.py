"""Thin wandb wrapper: metric/video logging with a monotonic-step guard.

wandb drops (with a warning) any log whose step is lower than one already
logged; the learner thread and the background replay thread both log, so the
guard here just clamps to the highest step seen instead of spamming warnings.

NOTE: this module was reconstructed from its call sites after the original
(untracked on BMRC — a bare `logs/` .gitignore pattern also matched `src/logs/`)
was lost in the Isambard migration.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

import wandb


class WandbLogger:
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
        dir: Optional[str] = None,
        tags=None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        id: Optional[str] = None,
        resume: Optional[str] = None,
    ):
        self._lock = threading.Lock()
        self._max_step = 0
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            mode=mode,
            dir=dir,
            tags=list(tags) if tags else None,
            notes=notes,
            group=group,
            job_type=job_type,
            id=id,
            resume=resume,
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        with self._lock:
            if step is not None:
                step = max(int(step), self._max_step)
                self._max_step = step
            self.run.log(metrics, step=step)

    def log_video(
        self,
        key: str,
        path: Path | str,
        fps: int = 15,
        step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        payload: Dict[str, Any] = {key: wandb.Video(str(path), format="mp4")}
        if extra:
            payload.update(extra)
        self.log(payload, step=step)

    def finish(self):
        try:
            self.run.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")
