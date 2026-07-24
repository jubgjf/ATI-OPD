#!/usr/bin/env bash
# Upload completed standalone OPD benchmark units to one stable W&B run per experiment.

python run/upload_opd_benchmark_wandb.py \
    --experiments dmpd--k3_0.15-20260716-180909 dmpd--train_on_teacher--loss_sft-20260721-085837 vanilla-opd-20260721-111103
