"""Collapse watchdog + auto-relaunch for the CNN-JEPA YOLO pretrain.

JEPA can silently collapse: the loss keeps dropping toward 0 while the target-encoder
features degenerate to a near-constant. The tell is `feature_std` (logged per epoch by the
model into the run's CSV) collapsing toward 0. This script:

  1. launches `pretrain/train_ijepa_yolo.py` as a subprocess (with the current lr),
  2. polls the run's metrics.csv for `train_metrics/feature_std`,
  3. CALIBRATES a collapse threshold from the first CALIBRATION_EPOCHS healthy epochs
     (threshold = COLLAPSE_FRAC * median(std over those epochs)),
  4. if std stays below the threshold for PATIENCE consecutive epochs -> declares a collapse,
     kills the run, halves the lr, and RELAUNCHES FROM SCRATCH (a collapsed checkpoint does
     not recover, so we do not resume from it),
  5. gives up after MAX_ATTEMPTS and prints a clear message for the human to intervene.

Usage (on the training box, inside the venv, from ~/CNN-JEPA):
    PYTHONPATH=. python scripts/watchdog.py
    PYTHONPATH=. python scripts/watchdog.py --lr 0.003 --max-attempts 4

The watchdog does NOT stop the whole thing automatically once it gives up — it exits and
leaves the decision to you, as requested.
"""
import argparse
import csv
import glob
import os
import signal
import subprocess
import sys
import time

# --- tunables -------------------------------------------------------------------------------
CALIBRATION_EPOCHS = 5     # healthy epochs used to establish the "normal" feature_std scale
COLLAPSE_FRAC = 0.3        # collapse threshold = COLLAPSE_FRAC * median(calibration stds)
PATIENCE = 3               # consecutive epochs below threshold before declaring collapse
POLL_SECONDS = 30          # how often to re-read the CSV
LR_DECAY = 0.5             # lr *= LR_DECAY on each relaunch
# Lightning suffixes on_epoch=True metrics with _epoch in the CSV (and adds a _step variant).
# We watch the per-epoch value.
STD_KEY = "train_metrics/feature_std_epoch"
# --------------------------------------------------------------------------------------------

ARTIFACTS_GLOB = "artifacts/pretrain_lightly/ijepacnn_yolo_maritime/*/version_*/csv/**/metrics.csv"


def _find_latest_csv(start_time):
    """Return the metrics.csv created by the run we just launched (newest mtime > start_time)."""
    candidates = [p for p in glob.glob(ARTIFACTS_GLOB, recursive=True)
                  if os.path.getmtime(p) >= start_time - 5]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _read_feature_std_series(csv_path):
    """All epoch-level feature_std values present in the CSV so far, in order."""
    out = []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                v = row.get(STD_KEY, "")
                if v not in ("", None):
                    try:
                        out.append(float(v))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return out


def _launch(lr, extra_overrides):
    """Start the training subprocess with the given lr. Returns the Popen handle."""
    cmd = [
        sys.executable, "pretrain/train_ijepa_yolo.py",
        "--config-name", "ijepacnn_yolo_maritime.yaml",
        f"optimizer.lr={lr}",
    ] + extra_overrides
    env = dict(os.environ, PYTHONPATH=".")
    print(f"\n[watchdog] launching:  {' '.join(cmd)}", flush=True)
    # start_new_session so we can kill the whole process group (dataloader workers included).
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def _kill(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_once(lr, extra_overrides):
    """Launch one training run and supervise it.

    Returns:
      "collapsed"  -> feature_std collapsed; caller should relaunch with a smaller lr
      "finished"   -> the run exited on its own (0 = done, !=0 = crashed; message printed)
    """
    start = time.time()
    proc = _launch(lr, extra_overrides)

    threshold = None
    below_streak = 0
    seen_epochs = 0

    while True:
        if proc.poll() is not None:
            code = proc.returncode
            if code == 0:
                print("[watchdog] training process exited cleanly (code 0).", flush=True)
                return "finished"
            # Non-zero exit = crash. code -9 (SIGKILL) is almost always the OOM killer
            # (too many dataloader workers / batch too large for RAM). Treat as a failure
            # the supervisor should relaunch, not a normal finish.
            reason = "OOM-killed (SIGKILL)" if code == -9 else f"crashed (code {code})"
            print(f"[watchdog] training process {reason}.", flush=True)
            return "crashed"

        time.sleep(POLL_SECONDS)
        csv_path = _find_latest_csv(start)
        if csv_path is None:
            continue
        stds = _read_feature_std_series(csv_path)
        if len(stds) <= seen_epochs:
            continue  # no new epoch since last poll
        seen_epochs = len(stds)

        # Calibration phase: establish the healthy scale, then derive the threshold.
        if threshold is None:
            if len(stds) >= CALIBRATION_EPOCHS:
                calib = sorted(stds[:CALIBRATION_EPOCHS])
                median = calib[len(calib) // 2]
                threshold = COLLAPSE_FRAC * median
                print(f"[watchdog] calibrated: healthy median std={median:.4f} over "
                      f"{CALIBRATION_EPOCHS} epochs -> collapse threshold={threshold:.4f}",
                      flush=True)
            else:
                print(f"[watchdog] calibrating... epoch {len(stds)}/{CALIBRATION_EPOCHS}, "
                      f"latest feature_std={stds[-1]:.4f}", flush=True)
            continue

        latest = stds[-1]
        if latest < threshold:
            below_streak += 1
            print(f"[watchdog] WARNING feature_std={latest:.4f} < {threshold:.4f} "
                  f"({below_streak}/{PATIENCE} epochs below)", flush=True)
            if below_streak >= PATIENCE:
                print(f"[watchdog] COLLAPSE confirmed (std below threshold for {PATIENCE} "
                      f"epochs). Killing run.", flush=True)
                _kill(proc)
                return "collapsed"
        else:
            if below_streak:
                print(f"[watchdog] recovered: feature_std={latest:.4f} >= {threshold:.4f}",
                      flush=True)
            below_streak = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.003, help="base learning rate")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("overrides", nargs="*",
                    help="extra hydra overrides, e.g. optimizer.batch_size=192")
    args = ap.parse_args()

    lr = args.lr
    collapse_attempts = 0
    crash_retries = 0
    MAX_CRASH_RETRIES = 2  # a crash (OOM/exception) is not a collapse: retry same lr a few times
    attempt = 0
    while collapse_attempts < args.max_attempts:
        attempt += 1
        print(f"\n===== [watchdog] attempt {attempt} (collapse {collapse_attempts + 1}/"
              f"{args.max_attempts})  lr={lr} =====", flush=True)
        result = run_once(lr, args.overrides)

        if result == "finished":
            print("[watchdog] run finished cleanly. Nothing more to do.", flush=True)
            return

        if result == "crashed":
            # Not a collapse — the lr is not the culprit. Retry the SAME lr a couple of times;
            # a crash on the very first epoch (before any feature_std) is usually OOM/config,
            # which relaunching identically won't fix, so cap the retries and then stop.
            crash_retries += 1
            if crash_retries > MAX_CRASH_RETRIES:
                print(f"\n[watchdog] GAVE UP: the run crashed {crash_retries} times at the same "
                      f"point (not a collapse). Likely OOM or a config/code error — check the "
                      f"log above. Fix (e.g. lower num_workers or batch_size) and relaunch. "
                      f"Not relaunching automatically.", flush=True)
                sys.exit(1)
            print(f"[watchdog] relaunching identically (crash retry {crash_retries}/"
                  f"{MAX_CRASH_RETRIES}); lr unchanged at {lr}.", flush=True)
            continue

        # result == "collapsed" -> relaunch from scratch with a smaller lr
        collapse_attempts += 1
        crash_retries = 0  # a real epoch ran, so reset the crash counter
        lr *= LR_DECAY
        if collapse_attempts < args.max_attempts:
            print(f"[watchdog] relaunching from scratch with lr={lr}", flush=True)

    print(f"\n[watchdog] GAVE UP after {args.max_attempts} collapse relaunches. The model keeps "
          f"collapsing even at lr={lr / LR_DECAY:.2e}. Human intervention needed: consider a "
          f"lower base lr, a longer lr warmup, a lower mask_ratio, or enabling the projection "
          f"head. Not relaunching automatically.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
