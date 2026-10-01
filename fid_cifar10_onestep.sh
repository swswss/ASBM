#!/bin/bash
# FID (50k samples, clean-fid legacy_tensorflow, CIFAR-10 train split) of a distilled one-step generator.
#   TEACHER=asbm (default): generator from distill_cifar10_asbm.sh
#   TEACHER=sm            : generator from distill_cifar10_sm.sh

TEACHER=${TEACHER:-asbm}
if [ "${TEACHER}" = "sm" ]; then
  CKPT=${CKPT:-checkpoint/sm_cifar10_distill/distill_1000ep.pt}
  ARGS="--beta-t1 20.0 --distill-cond-t 0.5"
else
  CKPT=${CKPT:-checkpoint/asbm_cifar10_distill/distill_1000ep.pt}
  ARGS="--beta-t1 4.0 --distill-cond-t 1.0"
fi

python main.py \
  --problem-name cifar10 \
  --eval-fid \
  --distill \
  --load ${CKPT} \
  --sde-type vp \
  --beta-t0 0.1 \
  ${ARGS} \
  --num-res-blocks-sm 4 \
  --distill-conv tweedie \
  --num-fid-samples 50000 \
  --fid-bs 1024 \
  --seed 917
