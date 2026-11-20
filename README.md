# Local residual memory for soil carbon spectroscopy

Research code for a soil organic carbon (SOC) study using the Open Soil Spectral Library (OSSL v1.2). Local residual memory (LRM) corrects the predictions of a fixed Cubist and a 1-D convolutional network. Each branch retrieves training residuals in its own representation, using PLS scores or CNN embeddings. The corrected branches are averaged on the log1p scale. The code uses the internal result name `SoilICL_lite` for LRM.

## Scope

The repository contains OSSL preprocessing, grouped splitting, model fitting, local reference insertion, the grouped-inner Cubist comparator, and frozen-prediction analyses. The analysis code covers cross-modality pairs, source enrichment, calibration readouts, agreement, correction magnitude, and neighbour-residual coherence. Raw or processed OSSL tables, coordinates, fitted weights, row-level predictions, and manuscript files are outside this code deposit.

## Data and environment

Use Python 3.12 and install the dependencies from the repository root.

```sh
python -m pip install -r requirements.txt
python pilot/scripts/verify_e1_summary.py
```

[The input manifest](analysis/ossl_v1_2_sources.json) records the four official download URLs and SHA-256 digests. Download them to `pilot/data/` and verify the checksums before using them. The [OSSL paper](https://doi.org/10.1371/journal.pone.0296545) provides the source description. Reuse is subject to the contributing datasets' licensing and access conditions. The scripts write arrays and metadata locally.

```sh
python pilot/scripts/00_prepare_mir_soc.py pilot/data
python pilot/scripts/12_prepare_visnir_soc.py pilot/data
```

## Fitting and split definitions

The experimental units are spectral records. Spatial folds group records by 1° cells where coordinates are available and use the study's site/source fallback otherwise. `pilot/scripts/p2_common.py` defines the grouped splits and the leave-source-out scenarios. Independently fitted MIR and VisNIR runs use separate spectral inputs.

### VisNIR spatial folds and source holdouts

For each scenario (`fold0` through `fold4`, `ICRAF`, and `KSSL`), fit the branches and generate predictions and, where applicable, local insertion trajectories. The following is the sequence for one complete fold. Repeat the sequence with the matching scenario.

```sh
python pilot/scripts/09a_p2_cubist.py --data visnir --scenario fold0
python pilot/scripts/09b_p2_cnn.py --data visnir --scenario fold0
python pilot/scripts/09c_p2_combine.py --data visnir --scenario fold0
python pilot/scripts/11_slim_p2_arrays.py --root V_visnir fold0
```

After reaching all five folds, run `python pilot/scripts/14_visnir_summary.py` to produce the pooled VisNIR spatial-fold table. Use the resulting saved queries to fit the stronger grouped-inner baseline. Repeat the first command for all five folds before aggregating.

```sh
python pilot/scripts/17_pls_cubist_baseline.py --data visnir --scenario fold0 --data-dir pilot/data --threads 4
python pilot/scripts/18_e1_summary.py --out-dir outputs/strong_cubist
```

### MIR development and confirmation folds

For each of the five MIR folds use the sequence below, running the stage-1 script for fold 0 first to write the common assignment. The summary distinguishes development fold 1 and confirmation folds 0, 2, 3 and 4.

```sh
python pilot/scripts/01_pilot_mir_soc.py --protocol P1 --fold 0
python pilot/scripts/03_stage2_hybrid.py --fold 0
python pilot/scripts/04_cnn_rrm.py --fold 0
# After repeating the sequence for all five folds:
python pilot/scripts/05_summarize_stage2.py
```

For MIR held-out source runs, use the branch-fitting, combination and slim-array commands from the VisNIR sequence with `--data mir` in the first three commands and `--root P2` in the last. Valid scenario names are `AfSIS`, `ICRAF`, `KSSL`, `LUCAS_Woodwell`, `SCHIEDUNG` and `SERBIA`. Generate the source mixtures with `python pilot/scripts/13_retrieval_provenance.py --data mir` while the full `cubist_side.npz` arrays still exist.

The fitting scripts write outputs under `pilot/results/`. Original experiments used separate software environments over the course of the study. Refitting compares with the frozen study only to the extent that grouped folds, preprocessing, parameters and predictions agree. In particular, CNN values may vary across package builds and hardware. Use the verified frozen outputs for numerical reproduction of the reported statistics.

## Analyse frozen predictions

The two scripts take explicit paths and write to a separate output directory. After generating the necessary inputs above, calculate the historical paired prediction comparison and prepare the evidence tables and row-level working inputs locally.

```sh
python pilot/scripts/15_visnir_mir_gap.py
python analysis/prepare_evidence.py --evidence outputs/paired_and_calibration
python analysis/analyse_transferable_error.py \
  --matched-predictions outputs/strong_cubist/matched_predictions_log1p.parquet \
  --paired-predictions outputs/paired_and_calibration/paired_predictions_all.parquet \
  --mir-phase outputs/paired_and_calibration/correction_phase_MIR.parquet \
  --output outputs/transferable_error
```

For an existing frozen run, point both analysis scripts to its `pilot/results` directory with `--pilot-results`. In addition to the supplied matched, paired and MIR phase tables, the transferable-error script reads the branch arrays and saved query predictions for the five VisNIR spatial folds, the VisNIR and MIR source holdouts, and the stored MIR confirmation predictions. It checks that the reconstructed LRM corrections match the stored predictions before evaluating the readouts. Paired group resampling describes uncertainty conditional on these predictions. The coherence score and branch agreement can be computed with the query spectrum and training-side residuals, while the analyses linking them to accuracy use the observed query outcomes retrospectively.

### Coherence of the applied correction and locality of residual information

`analysis/analyse_coherence.py` computes the coherence c = δ²/τ² of the applied LRM correction, the expected-benefit condition c > 2, label-free rankings (coherence, expected benefit, correction magnitude, neighbour closeness and spread), and residual memories restricted to same-source neighbour-rank bands or to other sources. It reads the VisNIR fold and source-holdout branch arrays and the VisNIR metadata (for reference-source labels) and refits nothing.

```sh
python analysis/analyse_coherence.py \
  --pilot-results pilot/results \
  --visnir-meta pilot/data/visnir_soc_meta.parquet \
  --output outputs/coherence
```

## Verification

The local frozen-input rerun reproduced 19 existing aggregate result tables to numerical tolerance. The grouped baseline's existing synthetic aggregation checks cover incomplete folds, changed query identities, and equivalence of the shared residual correction. The release check used existing frozen model outputs for the statistical analyses.
