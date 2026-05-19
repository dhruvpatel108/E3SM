Batch-aware TorchScript export prompt for the Deception-side agent
=================================================================

We have an existing PyTorch/TorchScript export path for a warm-rain emulator
used by E3SM/FTorch.

I need you to modify the exporter so it creates a new batch-aware TorchScript
model for FTorch integration.

Requirements
------------

- Keep the trained checkpoint and learned weights unchanged.
- Keep the same physical input feature order and output tendency order.
- Keep the same embedded preprocessing, postprocessing, QC/CLOUD masking, and
  any tail calibration logic.
- Export a model with interface:
  - input: `torch.Tensor` of shape `[11, N]`
  - output: `torch.Tensor` of shape `[4, N]`
- Internally, you may transpose to `[N, 11]`, run the existing vectorized
  model path, then transpose back.
- The implementation must support arbitrary batch size `N >= 1`.

Validation required before finishing
------------------------------------

1. `N=1` output from the new batch model must match the current scalar
   TorchScript model to tolerance.
2. For a random test batch, batch inference must match a scalar row-by-row
   loop over the current scalar model.
3. Rows failing the QC/CLOUD filter must return zero tendencies.
4. Save the new `.pt` file and any metadata, and report exact file paths.

Requested deliverables
----------------------

- updated exporter script
- saved batch-aware TorchScript `.pt`
- metadata `.json` if applicable
- a short validation summary with max absolute differences and tested shapes

Suggested smoke tests
---------------------

- verify input shapes `[11,1]`, `[11,8]`, and `[11,72]`
- verify output shape `[4,N]`
- include mixed active/inactive rows in the validation batch
