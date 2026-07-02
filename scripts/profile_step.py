"""Profile IJEPA_YOLO training steps on the GPU with a synthetic batch (no dataloader).

Measures the pure model step (student fwd, teacher fwd, loss, bwd, optimizer, EMA) and
prints throughput + the top CUDA ops. Run on the training box:

    PYTHONPATH=. python scripts/profile_step.py
    PYTHONPATH=. python scripts/profile_step.py optimizer.batch_size=128 perf.compile=false

Any hydra override works. The dataloader is bypassed on purpose: compare the img/s printed
here against the real training loop to tell a model-step bottleneck from a data-pipeline one.
"""
import sys
import time

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

INPUT_SIZE = 640  # matches the maritime config (trainer_common.input_sizes["maritime"])
WARMUP_STEPS = 8  # covers cudnn.benchmark autotuning + torch.compile warmup
TIMED_STEPS = 20
PROFILED_STEPS = 5


def main():
    with initialize(version_base="1.2", config_path="../pretrain/configs"):
        cfg = compose(config_name="ijepacnn_yolo_maritime.yaml", overrides=sys.argv[1:])
    OmegaConf.set_struct(cfg, False)
    cfg.wandb = False

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    from pretrain.train_ijepa_yolo import IJEPA_YOLO

    model = IJEPA_YOLO(cfg)
    model._setup_masking(INPUT_SIZE)
    model.log = lambda *a, **k: None  # self.log needs an attached pl.Trainer; profiling has none
    model = model.cuda().train()

    bs = int(cfg.optimizer.batch_size)
    x = torch.randn(bs, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.train_val_step((x, None, None), 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model._ema_update(0.996)
        return loss

    print(f"config: bs={bs} perf={OmegaConf.to_container(cfg.get('perf', {}))}", flush=True)
    print(f"warmup ({WARMUP_STEPS} steps; includes compile if enabled)...", flush=True)
    t0 = time.perf_counter()
    for _ in range(WARMUP_STEPS):
        one_step()
    torch.cuda.synchronize()
    print(f"warmup done in {time.perf_counter() - t0:.1f}s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        one_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"\n{TIMED_STEPS} steps in {dt:.2f}s -> {TIMED_STEPS / dt:.2f} it/s, "
          f"{TIMED_STEPS * bs / dt:.0f} img/s (model step only, no dataloader)")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILED_STEPS):
            one_step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    trace_path = "artifacts/profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"chrome trace -> {trace_path} (open in chrome://tracing or perfetto.dev)")


if __name__ == "__main__":
    main()
