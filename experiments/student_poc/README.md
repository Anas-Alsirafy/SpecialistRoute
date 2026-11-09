# Student PoC on Kaggle (teacher → student, Tomato)

This folder continues the data-difficulty experiment on Kaggle, now that the AUC GPU is gone.

**The question.** Some Tomato training images teach the model nothing useful: they are hard or uninformative, not mislabelled. If out-of-fold judges flag them and we train without them, does the student generalise better than:

- the teacher (all images), and
- a control that drops the same number of random images?

**What round 1 at AUC showed (seed 42 only):**

| Arm | Val log-loss | Correct / 57 |
|---|---|---|
| Teachers (5 seeds) | 0.8847 (mean) | — |
| C_remove | 0.8552 | 43 |
| R_random | 0.9269 | 40 |
| D_relabel | 0.9349 | 39 |

C beat the random control and D failed. That points to "difficult", not "mislabelled". It is one seed, though, so it needs five.

## Files

| Path | What it is |
|---|---|
| `kaggle_runner.ipynb` | The only notebook. Cell 1 = the job; cells 2–6 are fixed. |
| `results/folds.csv` per round | The 5 judge folds (written by the first judge job, then reused). |
| `results/<round>/judges/oof_fold{k}.npz` | Out-of-fold probabilities of judge *k* (`probs`, `rel`, `classes`). |
| `results/<round>/judges/fold{k}/result.json` | Judge held-out and val metrics, selected epoch, environment. |
| `results/<round>/flags.csv` | cleanlab label quality and `label_issue` per training image. |
| `results/probe.csv` | Frozen ConvNeXt-Large probe prediction per image (used by E_hybrid only). |
| `results/teachers/students/A_all_seed{s}/` | Kaggle teacher runs (all images). Shared by every round. |
| `results/<round>/students/<arm>_seed{s}/` | `result.json`, `history.csv`, and `test_probs_SEALED.npz`. |
| `results/<round>/summary_validation.csv`, `decision.json` | Written by the summary job. |
| `results/<round>/TEST_unsealed.csv` | Only after a decision is printed and you ask to unseal. |

The `rel` key is `<class folder>/<file name>`, so results do not depend on where the dataset is mounted.

## One-time setup (about 15 minutes)

1. **Dataset.**
   - On Kaggle, go to *Datasets → New Dataset* and upload `dataset-…zip`, the full one with `val` = 209 images.
   - Name it `plantdoc-v2` and keep it private.
   - Kaggle unzips it. The runner finds any folder called `dataset` that holds `train/`, `val/` and `test/`, so the exact name does not matter.
   - Optional: upload the reports zip as a second dataset. It is not needed by the runner.
2. **GitHub token.**
   - On GitHub, go to *Settings → Developer settings → Fine-grained tokens → Generate*.
   - Repository access: only `Anas-Alsirafy/SpecialistRoute`.
   - Permissions: *Contents: Read and write*.
   - Expiry: 90 days.
3. **Notebook.**
   - On Kaggle, go to *Create → New Notebook → File → Import Notebook* and upload `kaggle_runner.ipynb` from this folder.
   - In the right panel, set:
     - *Session options → Accelerator*: **GPU P100**, or T4. Pick one and keep the same GPU type for a whole round; `result.json` records it.
     - *Internet*: **On**, for the ImageNet weights and the git push.
     - *Add input* → your `plantdoc-v2` dataset.
     - *Add-ons → Secrets* → add `GITHUB_TOKEN` with the token, and tick it for this notebook.
4. **Selftest.**
   - Set `JOB = {"kind": "selftest"}` and press *Save Version → Save & Run All (Commit)*.
   - It takes about 3 minutes.
   - It must print `SELFTEST PASSED`. This proves the repo clone, the dataset, the GPU and the two-stage fine-tuning fix (backbone weights change only after unfreeze + recompile).

## Running a job

1. Copy one job line over `JOB` in cell 1.
2. Press *Save Version → Save & Run All (Commit)*. You can close the browser; commits run in the background.
3. When it finishes, the results are already in the repo. A commit message `student_poc: …` appears on GitHub.
4. If the push failed, the *Output* tab has `results_backup.zip`. Send it to me, or unzip it into `results/` and push.

Jobs skip work whose results are already in the repo. Re-running a finished job costs about 2 minutes, not hours.

## Run order: round 1 (60-epoch judges, reproduced on Kaggle)

| # | Job | GPU time (P100, rough) | Why |
|---|---|---|---|
| 0 | `selftest` | 3 min | Environment check |
| 1 | `student r1 A_all seed 42` | 4.2 h (T4, 300 epochs) | **Calibration — DONE 2026-11-06: 0.8465, 41/57, epoch 132 → PASS** |
| 2–7 | `judge r1 fold "all"` | ~2–2.5 h in one commit | 5 out-of-fold judges, each pushed when done; flags (cleanlab) run automatically at the end |
| 8 | `probe` | 10–15 min | Needed by E_hybrid only |
| 9 | `student r1 E_hybrid seed 42` | 2.5–4 h | Finishes the AUC seed-42 table |
| 10–14 | `student r1 C_remove` seeds 42, 123, 2024, 779, 1027 | 2.5–4 h each | Main arm |
| 15–19 | `student r1 R_random` same 5 seeds | 2.5–4 h each | Control |
| 20 | `summary r1` | 2 min | Prints the table and the decision |

Before step 1, the summary job reads the calibration result:

- **PASS:** the 5 AUC teacher losses stand in for A_all, so no more A_all runs are needed. Total is about **11 Large runs ≈ 30–45 GPU h**, about 1.5 weekly quotas.
- **FAIL:** Kaggle and AUC are not comparable. Also run A_all seeds 123, 2024, 779 and 1027 (+4 runs), and the decision uses only Kaggle numbers.

D_relabel is not repeated for five seeds, because it already lost clearly at AUC. Add it only if you want the full table for the paper.

**AUC judge files.** If you find `oof_fold0..4.npz` from the AUC PoC folder (`v2_TomatoPoC`):

1. Upload them as a Kaggle dataset.
2. Run `{"kind": "import_auc", "round": "r1"}` instead of steps 2–7, then run `flags r1`.

This saves about 2 GPU hours and keeps exactly the AUC flags. Likewise, if `Tomato_label_quality.csv` is attached, `probe` imports it instead of recomputing.

## Round 2 (genuinely fine-tuned judges)

Round 1 judges never really fine-tuned (the Keras 3 `FineTuneCallback` issue: setting `trainable` mid-fit without recompiling leaves the backbone frozen). Round 2 uses two-stage training:

- **Stage 1:** frozen backbone until val_loss stops improving for 25 epochs (max 100).
- **Stage 2:** unfreeze, recompile at 5e-5, and train until val_loss stops improving for 25 epochs (max 300).

Batch size is 16 so the trainable ConvNeXt-Base fits in 16 GB.

Jobs are the same with `"round": "r2"`: judges 0–4, `flags r2`, then C and R students. A_all and the probe are shared with round 1. Run round 2 only after round 1 is decided.

## Epoch budget (decided 2026-11-09, before any student result)

The calibration run peaked at epoch 132 (AUC teachers: 76–179), so every student run from now on uses **250 epochs** (`STUDENT_EPOCHS` in cell 2), about 3.5 h on a T4. Selection is by minimum val loss, so no result that peaks before 250 changes.

## Decision rule (pre-registered; also in cell 2)

- **Primary:** mean val log-loss of the arm < mean of A_all **and** < mean of R_random (5 seeds each).
- **Secondary:** mean val accuracy or macro-F1 > A_all mean.
- **Support:** at least 3 of the 5 arm seeds are below the A_all mean log-loss.
- **WIN** = all three.
- **Ties:** a difference of ≤ 1 validation image counts as a tie, and the simpler option is kept.

The test set (69 Tomato images) is predicted on every run but saved **sealed**. Unseal it once, after the decision: `{"kind": "summary", "round": "r1", "unseal_test": True}`.

## Notes

- **Code reuse.** The runner executes `notebooks/PlantDoc_Main.ipynb` sections 1–8 unchanged, except for the AUC paths. It uses the same loader, augmentation, models and multi-seed runner as the teachers.
- **Time limit.** Kaggle stops a session at 12 h. Training stops itself at 11 h and marks the run `truncated: true`; treat such a run as invalid.
- **Checkpoints.** They stay in the commit's *Output* (`/kaggle/working/Experiments`), not in the repo.
- **Token safety.** The repo clone lives in `/tmp`, so the token never appears in saved output.
- **Kaggle vs AUC.** The software differs (TensorFlow and Keras versions are in every `result.json`), and that is the reason for the calibration run.
