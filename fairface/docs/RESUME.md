# Resume this local task

Project root: `/Users/eliaswu/Desktop/Junior/Research/image-dataset-curation-october6`.
User authorized local FairFace dataset construction and a Google Slides update.
User did not authorize a GitHub push, training, paid API calls, or Dots processing.

1. Read `runtime/progress.json`, `runtime/run.log`, `data/selection_report.json`
   (if present), and `slides/upload_verification.json` (if present).
2. Check whether another writer is running. `runtime/writer.lock` is an OS lock,
   not a PID marker. Do not delete it; the pipeline rejects concurrent writers.
3. Run `bash scripts/run_local.sh run` to continue source screening/inference and
   matching. Committed batches are reused; partial batches can repeat. Do not edit
   files under `src/` or the frozen config during a run. The checkpoint contract
   includes their hashes and intentionally rejects drift. Source inputs remain
   in historical folders, read-only; local paths are in `runtime/local_paths.json`.
4. Wait for `data/DATASET_READY.json`, then run `bash scripts/run_local.sh validate`.
5. Using `../image-dataset-curation-september29/.venv/bin/python -B`, run:
   `scripts/reproduce.py --rebuild-images`, `scripts/audit_release.py`,
   `scripts/analyze.py`, and `scripts/package.py`.
6. Run unit tests and Ruff. Review all six local plots. If improving labels or
   documentation, edit `scripts/` or docs, not the core checkpoint code.
7. Inspect the existing shared deck before updating it using
   `scripts/slides_update.py inspect`. Its service-account authentication can
   require approved access to the user's existing gcloud credential store.
   `scripts/slides_update.py upload` creates six native editable charts with
   separate titles/captions, preserves historical slide content/order, and copies
   the original name/date geometry. It refuses to create duplicates if the
   October 6 IDs already exist. If upload was interrupted, inspect live IDs first;
   do not blindly rerun or delete slides.
8. Visually inspect every new slide thumbnail under `runtime/slides/` and verify
   the remote header size/position and original slides. Fix only owned October 6
   objects if necessary. Do not modify historical slides or user-edited titles.
9. Rerun packaging/checksums after all final code/docs/slide records are saved.
   Verify `SHA256SUMS` and ZIP CRC. Keep face images private and leave GitHub
   untouched until separate approval.

Shared presentation:
https://docs.google.com/presentation/d/1pbxNh3HMsDw4BKXKuVy4UHZloxpPal_ZMcGblqm7bc4/edit

Final response should report actual pairs/images, train/dev/test counts, validation,
uncertainty/limitations, local paths, the slide link, and zero model-API/GPU spend.
Never describe image matching as a DPO chosen/rejected dataset or a debiasing result.
