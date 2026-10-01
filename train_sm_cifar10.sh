#!/bin/bash
# Score-matching baseline on CIFAR-10: denoising score matching on the (memoryless) VP SDE with beta: 0.1 -> 20.
# This model is the teacher of distill_cifar10_sm.sh. Checkpoints go to checkpoint/sm_cifar10.

python main.py \
  --problem-name cifar10 \
  --name sm_cifar10 \
  --data-root data \
  --score-matching \
  --sde-type vp \
  --beta-t0 0.1 \
  --beta-t1 20.0 \
  --num-res-blocks-sm 4 \
  --backward-epochs 3300 \
  --sm-bs 128 \
  --lr-sm 2e-4 \
  --sample-nfe 100 \
  --log-tb

# FID of the score-matching model (uniform steps; SAMPLER=euler: reverse SDE, heun: probability-flow ODE):
#   python main.py --problem-name cifar10 --score-matching --sde-type vp --beta-t0 0.1 --beta-t1 20.0 \
#     --eval-fid --load checkpoint/sm_cifar10/backward_3300ep.pt --sampler euler --sample-nfe 100
