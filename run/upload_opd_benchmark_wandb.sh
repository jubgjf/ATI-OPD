#!/usr/bin/env bash
# Upload completed standalone OPD benchmark units to one stable W&B run per experiment.

python -m verl.utils.evaluation.opd_benchmark_wandb \
    --experiments vanilla-opd-20260716-122946