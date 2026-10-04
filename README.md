<p align="center">
<h1 align="center">ASBM (ICML 2026)</h1>
<h3 align="center">Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schrödinger Bridge Matching</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://swswss.github.io/">Jeongwoo Shin</a><sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?user=kmXfNc4AAAAJ&hl=en">Jinhwan Sul</a><sup>3</sup>
    ·
    <a href="http://www.joonseok.net/home.html">Joonseok Lee</a><sup>1†</sup><br>
    <a href="https://sites.google.com/view/jaewoongchoi/home">Jaewoong Choi</a><sup>2†</sup>
    ·
    <a href="https://jaemoo-choi.github.io/">Jaemoo Choi</a><sup>3†</sup><br>
    <sup>1</sup>Seoul National University <sup>2</sup>Sungkyunkwan University <sup>3</sup>Georgia Institute of Technology<br>
    <sup>†</sup>Corresponding author
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2602.15396">Paper</a></h3>
</p>

---

Official PyTorch implementation of **ASBM** ([arXiv:2602.15396](https://arxiv.org/abs/2602.15396)), with training and evaluation code for CIFAR-10 (pixel space) and FFHQ-256 (latent space), the score-matching baseline, and one-step distillation.

ASBM learns a generative model by solving the Schrödinger bridge problem, a dynamic (entropy-regularized) optimal transport problem between the data distribution and a Gaussian prior.

ASBM training has two stages, which a single command runs in sequence:

1. **Forward process.** Learns the forward control of a Schrödinger bridge between data (t=0) and a Gaussian prior (t=1), relative to a reference SDE (VP by default). It alternates *adjoint matching* for the control with *corrector matching* for the terminal corrector. The forward model is selected by checking that its terminal samples match the prior (covariance spectrum and radial statistics).
2. **Backward process.** Learns the backward score by bridge score matching on couplings `(x0, x1)` simulated with the learned forward process.

Samples are generated from the prior by integrating the learned backward SDE (Euler–Maruyama) or the probability-flow ODE (Heun) with uniform time steps.

## Installation

```bash
conda env create -f environment.yaml
conda activate asbm
```

## Pre-trained checkpoints

| Dataset | File | Contents |
|---|---|---|
| CIFAR-10 (32×32) | `checkpoints/asbm_cifar10.pt` | EMA forward control + EMA backward score |
| FFHQ-256 (latent 4×32×32) | `checkpoints/asbm_ffhq_ldm.pt` | EMA forward control + EMA backward score |

The checkpoints are stored with Git LFS. If `git clone` gave you small pointer files instead, run `git lfs pull`.

## Data

- **CIFAR-10** is downloaded automatically to `--data-root` (default: `data/`).
- **FFHQ-256**: training uses pre-computed latents from the [`stabilityai/sd-vae-ft-ema`](https://huggingface.co/stabilityai/sd-vae-ft-ema) VAE. Download [FFHQ](https://github.com/NVlabs/ffhq-dataset) (`images1024x1024`) and run:
  ```bash
  python prepare_ffhq_latents.py --image-dir /path/to/ffhq/images1024x1024 --out-dir data/ffhq_latents
  ```
  This resizes each image to 256×256 and stores its VAE posterior `[mean; std]` as one `.npy` file per image.

## Training

```bash
bash train_cifar10.sh
LATENT_DIR=data/ffhq_latents bash train_ffhq_ldm.sh
```

Checkpoints (`forward_*.pt`, `forward_best.pt`, `backward_*ep.pt`), logs (`train_log.txt`, TensorBoard) and sample grids go to `checkpoint/<name>/`.
- To resume the backward stage, add `--start-from-sm --load checkpoint/<name>/backward_<N>ep.pt`.
- To start the backward stage from a forward checkpoint, add `--start-from-sm --reinit-score-model --load <forward checkpoint>`.

## Evaluation (FID)

```bash
bash fid_cifar10.sh      # 50k samples, clean-fid (legacy_tensorflow), CIFAR-10 train split
bash fid_ffhq_ldm.sh     # 70k samples, clean-fid (legacy_tensorflow), FFHQ trainval70k split
```

You can override the sampler with environment variables, e.g. `SAMPLER=heun SAMPLE_NFE=50 bash fid_cifar10.sh`.
- `SAMPLER`: `euler` (backward SDE) or `heun` (probability-flow ODE).
- `SAMPLE_NFE`: number of uniform steps from t=1 to t=0. The last step is noise-free.

Generated images are written next to the checkpoint (`samples_*`), or to `--gen-dir` if given.

## Score-matching baseline

Denoising score matching on the memoryless VP SDE (β from 0.1 to 20), with the same UNet as the ASBM backward network:

```bash
bash train_sm_cifar10.sh
```

The end of the script shows how to evaluate it: `--score-matching --eval-fid`, with the same samplers as above.

## One-step distillation (CIFAR-10)

The teacher (ASBM, or the score-matching baseline) is distilled into a one-step generator. Training combines a distribution-matching loss, with a fake score trained online on the generator's samples, and a regression loss to (noise, teacher ODE sample) pairs.

```bash
bash distill_cifar10_asbm.sh              # teacher: checkpoints/asbm_cifar10.pt
bash distill_cifar10_sm.sh                # teacher: checkpoint/sm_cifar10/backward_3300ep.pt
bash fid_cifar10_onestep.sh               # FID of the ASBM-distilled generator
TEACHER=sm bash fid_cifar10_onestep.sh    # FID of the SM-distilled generator
```

The 500k regression pairs (about 12GB) are generated once with the teacher and reused. They are stored in the run folder unless `--distill-pair-path` is given.

## Main options

The defaults in `options.py` are the paper settings for ASBM: a VP reference SDE with β from 0.1 to 4.0, an N(0, I) prior, and corrector matching as the first stage. Other options:

| Option | Values |
|---|---|
| `--sde-type` | `vp` (`--beta-t0`, `--beta-t1`, `--vp-sigma`), `vp_const`, `ve` / `edm_ve` (`--sigma-min`, `--sigma-max`), `linear` / `linear_std` (`--linear-m`, `--linear-k`), `brownian` (`--linear-k`) |
| `--flow-matching` | backward process parametrized as a flow-matching ODE instead of a score |
| `--prior-std` | std of the Gaussian prior |
| `--init-stage` | `bridge` (corrector matching first) or `adjoint` |
| `--forward-am-epochs`, `--forward-cm-epochs` | epochs per adjoint- / corrector-matching stage |
| `--forward-nfe`, `--cm-nfe` | discretization of the forward SDE in the adjoint / corrector stages |
| `--score-matching` | score-matching baseline instead of ASBM |
| `--distill-*` | one-step distillation settings |

Run `python main.py --help` for the full list.

## Citation

```bibtex
@inproceedings{shin2026efficient,
  title={Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schr{\"o}dinger Bridge Matching},
  author={Shin, Jeongwoo and Sul, Jinhwan and Lee, Joonseok and Choi, Jaewong and Choi, Jaemoo},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgements

This code builds on [SB-FBSDE](https://github.com/ghliu/SB-FBSDE), [Adjoint Sampling / ASBS](https://github.com/facebookresearch/adjoint_samplers), [torchcfm](https://github.com/atong01/conditional-flow-matching) and [guided-diffusion](https://github.com/openai/guided-diffusion) (UNet), [DMD2](https://github.com/tianweiy/DMD2) (distillation), and [clean-fid](https://github.com/GaParmar/clean-fid) (FID).
