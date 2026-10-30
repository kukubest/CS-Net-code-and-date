# Reproducibility

## Experimental roles

- Existing training partition: training only.
- Historical `test` partition: validation/checkpoint selection only.
- PCB 6 independent set: final testing only; it is never used for training,
  validation, early stopping, hyperparameter selection, or epoch selection.
- PCB 7 and PCB 8: additional cross-board generalization evaluation.

## Volume convention

Each sample is float32 with shape `40 x 256 x 256`. The common strategy pads
the Z axis reflectively to 64 slices, performs the network operation, and crops
back to the real 40-slice region. Metrics are computed on the valid 40 slices.

The recorded PCB 6 independent test set contains 200 paired GT/SIRT samples,
generated with seed 20260916. Expected RAW size is 10,485,760 bytes per file.

## Five ablation groups

- A `swin_unetr_baseline`: original Swin-UNETR, L1.
- B `csnet_no_compression`: CS-Net modules retained, compression bypassed, L1.
- C `csnet_compressed_no_hr_skip`: compressed backbone without HR skip, L1.
- D `csnet_l1`: full CS-Net, L1.
- E `csnet_mse`: full CS-Net, MSE.

The causal comparisons are B vs D (compression), C vs D (HR skip), D vs E
(loss), and A vs D (accuracy/efficiency trade-off).

## Timing and efficiency

Inference timing uses 50 warm-up forwards and 100 timed forwards with CUDA
synchronization. The result JSON files document the FLOPs and peak-VRAM scope.

## Determinism

The runner records Python, PyTorch, CUDA, cuDNN, GPU, seed, and deterministic
settings. The recorded configuration uses deterministic algorithms with
`warn_only=True`; warnings should be retained because some GPU operators may
not have fully deterministic implementations.
