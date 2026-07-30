#!/usr/bin/env bash
set -e

python -m sequence_leakage_audit.overlap_diagnostic \
  --manifest experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv \
  --split-col split_random \
  --train-name train \
  --eval-names test \
  --input-col seq_row_ids \
  --target-id-col target_row_id \
  --sequence-id-col sequence_id \
  --thresholds 1,3,5,6 \
  --out-dir experiments/traqid_pretraining_v1/reports/sequence_overlap_diagnostics/random_split_T7
