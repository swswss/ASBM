#!/bin/bash
# Latent ASBM on FFHQ-256 (SD-VAE latents, 4x32x32; VP reference process with beta: 0.1 -> 4.0).
# Pre-compute the latents first:
#   python prepare_ffhq_latents.py --image-dir /path/to/ffhq/images1024x1024 --out-dir data/ffhq_latents
# Checkpoints, logs and sample grids are written to checkpoint/asbm_ffhq_ldm.
# The released checkpoint (checkpoints/asbm_ffhq_ldm.pt) is the EMA model after 600 score-matching epochs.

LATENT_DIR=${LATENT_DIR:-data/ffhq_latents}

python main.py \
  --problem-name ffhq \
  --name asbm_ffhq_ldm \
  --dataset-path ${LATENT_DIR} \
  --sde-type vp \
  --beta-t0 0.1 \
  --beta-t1 4.0 \
  --prior-std 1.0 \
  --init-stage bridge \
  --forward-epochs 1000 \
  --forward-am-epochs 12 \
  --forward-cm-epochs 8 \
  --forward-nfe 50 \
  --cm-nfe 100 \
  --buffer-size 20000 \
  --forward-resample-size 4096 \
  --asbs-bs 128 \
  --cm-bs 128 \
  --lr-f 1e-4 \
  --lr-b 1e-4 \
  --num-res-blocks 2 \
  --num-res-blocks-sm 4 \
  --backward-epochs 600 \
  --backward-buffer-size 70000 \
  --backward-resample-size 70000 \
  --sm-bs 128 \
  --lr-sm 1e-4 \
  --samp-bs 512 \
  --sample-nfe 100 \
  --log-tb

# To resume stage 2 from a saved checkpoint:
#   python main.py --problem-name ffhq [same options] --start-from-sm --load checkpoint/asbm_ffhq_ldm/backward_<N>ep.pt
# To run stage 2 only, starting from a stage-1 checkpoint:
#   python main.py --problem-name ffhq [same options] --start-from-sm --reinit-score-model --load checkpoint/asbm_ffhq_ldm/forward_adjoint_1000ep.pt
