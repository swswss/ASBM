#!/bin/bash
# Generate 50k CIFAR-10 samples with the released checkpoint and compute FID (clean-fid, legacy_tensorflow, train split).
# Sampler: Euler-Maruyama on the learned backward SDE with SAMPLE_NFE uniform steps from t=1 to t=0
# (the last step is noise-free). Use SAMPLER=heun for the probability-flow ODE (Heun).

CKPT=${CKPT:-checkpoints/asbm_cifar10.pt}
SAMPLER=${SAMPLER:-euler}
SAMPLE_NFE=${SAMPLE_NFE:-100}

python main.py \
  --problem-name cifar10 \
  --eval-fid \
  --load ${CKPT} \
  --sde-type vp \
  --beta-t0 0.1 \
  --beta-t1 4.0 \
  --prior-std 1.0 \
  --num-res-blocks 2 \
  --num-res-blocks-sm 4 \
  --sampler ${SAMPLER} \
  --sample-nfe ${SAMPLE_NFE} \
  --num-fid-samples 50000 \
  --fid-bs 512 \
  --seed 917
