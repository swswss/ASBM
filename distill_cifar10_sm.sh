#!/bin/bash
# One-step distillation of the score-matching baseline (teacher: train_sm_cifar10.sh, VP beta: 0.1 -> 20).
# Distribution matching on the VP marginal + regression to 500k teacher ODE (Heun, 50 steps) pairs.
# The pairs (~12GB) are generated once into the run folder (or --distill-pair-path) and reused.
# Checkpoints go to checkpoint/sm_cifar10_distill; evaluate with fid_cifar10_onestep.sh (TEACHER=sm).

TEACHER=${TEACHER:-checkpoint/sm_cifar10/backward_3300ep.pt}

python main.py \
  --problem-name cifar10 \
  --name sm_cifar10_distill \
  --data-root data \
  --distill \
  --score-matching \
  --load ${TEACHER} \
  --sde-type vp \
  --beta-t0 0.1 \
  --beta-t1 20.0 \
  --num-res-blocks-sm 4 \
  --distill-conv tweedie \
  --distill-cond-t 0.5 \
  --distill-lambda-dm 1.0 \
  --distill-lambda-reg 0.1 \
  --distill-reg-nfe 50 \
  --distill-num-pairs 500000 \
  --distill-lr 1e-5 \
  --distill-update-ratio 5 \
  --distill-bs 128 \
  --distill-epochs 1000
