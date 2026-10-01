#!/bin/bash
# One-step distillation of the ASBM CIFAR-10 model (teacher: checkpoints/asbm_cifar10.pt).
# Distribution matching on the reference bridge + regression to 500k teacher ODE (Heun, 50 steps) pairs.
# The pairs (~12GB) are generated once into the run folder (or --distill-pair-path) and reused.
# Checkpoints go to checkpoint/asbm_cifar10_distill; evaluate with fid_cifar10_onestep.sh.

TEACHER=${TEACHER:-checkpoints/asbm_cifar10.pt}

python main.py \
  --problem-name cifar10 \
  --name asbm_cifar10_distill \
  --data-root data \
  --distill \
  --load ${TEACHER} \
  --sde-type vp \
  --beta-t0 0.1 \
  --beta-t1 4.0 \
  --num-res-blocks 2 \
  --num-res-blocks-sm 4 \
  --distill-conv tweedie \
  --distill-cond-t 1.0 \
  --distill-weight bt_half \
  --distill-w-scaler 0.9 \
  --distill-lambda-dm 1.0 \
  --distill-lambda-reg 0.1 \
  --distill-reg-nfe 50 \
  --distill-num-pairs 500000 \
  --distill-lr 1e-5 \
  --distill-update-ratio 5 \
  --distill-bs 128 \
  --distill-epochs 1000
