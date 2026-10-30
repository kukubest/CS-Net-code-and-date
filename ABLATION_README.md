# CS-Net ablation framework

The unified entry point is `run_ablation.py`; all shared settings and all five
experiment definitions live in `ablation_config.json`.

## Data roles

- `Final_Adaptive_*/train`: optimization only.
- legacy `Final_Adaptive_*/test`: validation/checkpoint selection only.
- `Final_Adaptive_*/independent_test`: final evaluation only. It is validated
  before paper metrics are computed and is never loaded by the training loop.

The physical volume is `[1, 40, 256, 256]`. Every model receives a Z-reflection
padded `[1, 64, 256, 256]` tensor; loss and metrics use only slices `12:52`.

## Commands

Run all five experiments (automatic resume is the default):

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --experiments all
```

Run one group:

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --experiments csnet_l1
```

Resume selected groups (same command; completed runs are skipped):

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --experiments csnet_compressed_no_hr_skip csnet_l1
```

Evaluate best validation checkpoints only:

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --mode eval-only --experiments all
```

Rebuild summaries/figures without training or inference:

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --mode summary-only
```

Run multiple seeds (recommended first for C/D):

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --experiments csnet_compressed_no_hr_skip csnet_l1 --seeds 20260916 20260917 20260918
```

Run the non-training smoke suite:

```powershell
G:\CT_RE\conda\envs\ct_pnp\python.exe run_ablation.py --mode smoke-test
```

## Recovery and safety

Each seed has `status.json`, `last.pt`, and `best_val.pt`. `last.pt` contains
model, optimizer, scheduler, AMP scaler, epoch, best validation loss, Python /
NumPy / Torch / CUDA RNG state, and a configuration summary. An epoch-zero
checkpoint is created before training so even a first-epoch interruption has a
clean recovery point. Re-running the same command resumes `last.pt`, skips
completed runs, and does not overwrite paper results. A failure in one group is
recorded and the controller continues unless `--fail-fast` is passed.

## Efficiency definitions

- Params: all registered model parameters. B retains the compression modules
  for exact parameter-set parity with D even though its forward pass bypasses
  them.
- FLOPs: two times THOP MACs for one executed forward on the padded full input.
- Peak VRAM: peak CUDA allocated bytes during one AMP forward, after resetting
  peak statistics; it excludes unrelated process allocations.
- Time: one-sample AMP forward, 50 warm-ups and 100 synchronized timed runs;
  mean and population standard deviation are reported.

Copper Dice/IoU is enabled because the project generator provides a verified
material label (`copper=1.00`, `solder=0.82`). GT copper uses exact label 1.00;
prediction threshold 0.91 is the midpoint. It is not an inferred GT threshold.
