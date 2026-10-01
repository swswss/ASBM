#!/bin/bash
# ASBM on CIFAR-10 (VP reference process with beta: 0.1 -> 4.0).
# Stage 1 (forward, adjoint/corrector matching) and stage 2 (backward, score matching) run in sequence.
# Checkpoints, logs and sample grids are written to checkpoint/asbm_cifar10.
# The released checkpoint (checkpoints/asbm_cifar10.pt) is the EMA model after 600 score-matching epochs.

python main.py \
  --problem-name cifar10 \
  --name asbm_cifar10 \
  --data-root data \
  --sde-type vp \
  --beta-t0 0.1 \
  --beta-t1 4.0 \
  --prior-std 1.0 \
  --init-stage bridge \
  --forward-epochs 1000 \
  --forward-am-epochs 15 \
  --forward-cm-epochs 5 \
  --forward-nfe 20 \
  --cm-nfe 100 \
  --buffer-size 20000 \
  --forward-resample-size 4096 \
  --asbs-bs 128 \
  --cm-bs 128 \
  --lr-f 1e-4 \
  --lr-b 1e-4 \
  --reset-optimizer \
  --num-res-blocks 2 \
  --num-res-blocks-sm 4 \
  --backward-epochs 600 \
  --backward-buffer-size 50000 \
  --backward-resample-size 50000 \
  --sm-bs 128 \
  --lr-sm 2e-4 \
  --samp-bs 512 \
  --sample-nfe 100 \
  --log-tb

# To resume stage 2 from a saved checkpoint:
#   python main.py --problem-name cifar10 [same options] --start-from-sm --load checkpoint/asbm_cifar10/backward_<N>ep.pt
# To run stage 2 only, starting from a stage-1 checkpoint:
#   python main.py --problem-name cifar10 [same options] --start-from-sm --reinit-score-model --load checkpoint/asbm_cifar10/forward_best.pt
