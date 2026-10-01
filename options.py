import argparse
import json
import os
import random
import sys

import numpy as np
import torch

import util

BOOL = argparse.BooleanOptionalAction


def build_parser():
    parser = argparse.ArgumentParser(description="ASBM: training / sampling / FID evaluation")

    # --------------- basic ---------------
    parser.add_argument("--problem-name", type=str, required=True, choices=["cifar10", "ffhq"])
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--name", type=str, default=None, help="experiment name (default: built from the hyperparameters)")
    parser.add_argument("--ckpt-root", type=str, default="checkpoint", help="experiments are saved under <ckpt-root>/<name>")
    parser.add_argument("--load", type=str, default=None, help="checkpoint to load (resume training / evaluation)")
    parser.add_argument("--log-tb", action="store_true", help="log losses with tensorboard")
    parser.add_argument("--data-root", type=str, default="data", help="[cifar10] download / load directory")
    parser.add_argument("--dataset-path", type=str, default=None, help="[ffhq] folder of pre-computed VAE latents")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="[ffhq] VAE for the latent space")

    # --------------- networks ---------------
    parser.add_argument("--num-res-blocks", type=int, default=2, help="res-blocks of the forward control and the corrector")
    parser.add_argument("--num-res-blocks-sm", type=int, default=4, help="res-blocks of the backward score network")

    # --------------- reference process & prior ---------------
    parser.add_argument("--sde-type", type=str, default="vp",
                        choices=["vp", "vp_const", "ve", "edm_ve", "linear", "linear_std", "brownian"])
    parser.add_argument("--beta-t0", type=float, default=0.1, help="[vp] beta at t=0 (data)")
    parser.add_argument("--beta-t1", type=float, default=4.0, help="[vp] beta at t=1 (prior)")
    parser.add_argument("--vp-sigma", type=float, default=1.0, help="[vp] diffusion scale")
    parser.add_argument("--sigma-min", type=float, default=0.01, help="[ve, edm_ve] sigma at t=0")
    parser.add_argument("--sigma-max", type=float, default=1.0, help="[ve, edm_ve] sigma at t=1")
    parser.add_argument("--edm-sigma", action="store_true", help="[edm_ve] use the EDM rho-schedule for sigma_t")
    parser.add_argument("--linear-m", type=float, default=0.6412035, help="[linear, linear_std] mean decay")
    parser.add_argument("--linear-k", type=float, default=0.9334158, help="[linear, linear_std, brownian] diffusion scale")
    parser.add_argument("--prior-std", type=float, default=1.0, help="std of the Gaussian prior at t=1")
    parser.add_argument("--flow-matching", action="store_true", help="parametrize the backward process as a flow-matching ODE")

    # --------------- stage 1: forward process (adjoint / corrector matching) ---------------
    parser.add_argument("--init-stage", type=str, default="bridge", choices=["bridge", "adjoint"],
                        help="first stage of the alternating forward training")
    parser.add_argument("--forward-epochs", type=int, default=1000)
    parser.add_argument("--forward-am-epochs", type=int, default=15, help="epochs per adjoint-matching stage")
    parser.add_argument("--forward-cm-epochs", type=int, default=5, help="epochs per corrector-matching stage")
    parser.add_argument("--forward-nfe", type=int, default=20, help="discretization of the forward SDE")
    parser.add_argument("--cm-nfe", type=int, default=100, help="forward-SDE discretization in corrector stages")
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--forward-resample-size", type=int, default=4096, help="new samples simulated per epoch")
    parser.add_argument("--dataloader-duplicate", type=int, default=1)
    parser.add_argument("--asbs-bs", type=int, default=128, help="batch size of adjoint matching")
    parser.add_argument("--cm-bs", type=int, default=128, help="batch size of corrector matching")
    parser.add_argument("--lr-f", type=float, default=1e-4, help="lr of the forward control")
    parser.add_argument("--lr-b", type=float, default=1e-4, help="lr of the corrector")
    parser.add_argument("--reset-buffer", action=BOOL, default=True, help="empty the buffers when the stage switches")
    parser.add_argument("--reset-optimizer", action=BOOL, default=False, help="re-build the optimizers when the stage switches")
    parser.add_argument("--simulate-ema", action=BOOL, default=True, help="simulate the forward SDE with the EMA control")
    parser.add_argument("--fwd-zero-noise", action="store_true", help="no noise at the last forward step")
    parser.add_argument("--do-forward-eval", action=BOOL, default=True,
                        help="check the terminal marginal (covariance spectrum, radial stats) to select the forward model")
    parser.add_argument("--forward-eval-start", type=int, default=300, help="first epoch to evaluate the forward model")

    # --------------- stage 2: backward process (score matching on the learned coupling) ---------------
    parser.add_argument("--backward-epochs", type=int, default=3300)
    parser.add_argument("--backward-buffer-size", type=int, default=50000)
    parser.add_argument("--backward-resample-size", type=int, default=50000, help="new couplings simulated per epoch")
    parser.add_argument("--sm-bs", type=int, default=128, help="batch size of score matching")
    parser.add_argument("--lr-sm", type=float, default=2e-4, help="lr of the score network")
    parser.add_argument("--backward-save-freq", type=int, default=10, help="save / preview every N epochs")
    parser.add_argument("--start-from-sm", action="store_true", help="skip stage 1 (requires --load)")
    parser.add_argument("--reinit-score-model", action="store_true",
                        help="with --start-from-sm: start the score network from scratch (--load is a stage-1 checkpoint)")

    # --------------- common training ---------------
    parser.add_argument("--samp-bs", type=int, default=512, help="batch size of all simulations")
    parser.add_argument("--ema-decay-fwd", type=float, default=0.999, help="EMA decay in stage 1")
    parser.add_argument("--ema-decay-bwd", type=float, default=0.999, help="EMA decay in stage 2")
    parser.add_argument("--l2-norm", type=float, default=0.0, help="weight decay")

    # --------------- sampling & evaluation ---------------
    parser.add_argument("--eval-fid", action="store_true", help="generate samples from --load and compute FID")
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun"],
                        help="euler: Euler-Maruyama on the backward SDE, heun: Heun on the probability-flow ODE")
    parser.add_argument("--sample-nfe", type=int, default=100, help="number of uniform sampling steps from t=1 to t=0")
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--fid-bs", type=int, default=512, help="batch size for generation")
    parser.add_argument("--gen-dir", type=str, default=None, help="where to save generated images")

    # --------------- score-matching baseline (memoryless VP diffusion, no forward process) ---------------
    parser.add_argument("--score-matching", action="store_true",
                        help="train / evaluate / distill the denoising score-matching baseline on the VP reference SDE")

    # --------------- one-step distillation (CIFAR-10) ---------------
    parser.add_argument("--distill", action="store_true",
                        help="distill the teacher (--load; ASBM, or the SM baseline with --score-matching) into a one-step "
                             "generator. With --eval-fid, --load is a distilled checkpoint and FID is computed for 1-step samples")
    parser.add_argument("--distill-epochs", type=int, default=1000)
    parser.add_argument("--distill-bs", type=int, default=128)
    parser.add_argument("--distill-lr", type=float, default=1e-5)
    parser.add_argument("--distill-update-ratio", type=int, default=5, help="generator update every N fake-score updates")
    parser.add_argument("--distill-conv", type=str, default="tweedie", choices=["tweedie", "vp", "scratch"],
                        help="how the generator maps the network output to x0")
    parser.add_argument("--distill-cond-t", type=float, default=1.0, help="time fed to the generator network")
    parser.add_argument("--distill-weight", type=str, default="bt_half", choices=["one", "bt", "bt_half"],
                        help="[ASBM teacher] weight of the distribution-matching gradient")
    parser.add_argument("--distill-w-scaler", type=float, default=0.9, help="[ASBM teacher] w = s * b_t + (1 - s) for bt_half")
    parser.add_argument("--distill-lambda-dm", type=float, default=1.0, help="weight of the distribution-matching loss")
    parser.add_argument("--distill-lambda-reg", type=float, default=0.1, help="weight of the regression loss (0: off)")
    parser.add_argument("--distill-reg-nfe", type=int, default=50, help="teacher ODE steps for the regression pairs")
    parser.add_argument("--distill-num-pairs", type=int, default=500000, help="number of (noise, teacher sample) pairs")
    parser.add_argument("--distill-pair-path", type=str, default=None, help="where to store the pairs (default: in the run folder)")
    parser.add_argument("--distill-save-freq", type=int, default=50, help="save the generator every N epochs")
    return parser


def default_name(opt):
    if opt.sde_type in ["vp", "vp_const"]:
        sde_arg = f"beta1_{opt.beta_t1}"
    elif opt.sde_type in ["ve", "edm_ve"]:
        sde_arg = f"sigmamax_{opt.sigma_max}"
    else:
        sde_arg = f"k_{opt.linear_k}_m_{opt.linear_m}"
    fm = "_fm" if opt.flow_matching else ""
    if opt.score_matching:
        name = f"sm_{opt.problem_name}_{opt.sde_type}_{sde_arg}"
    else:
        name = f"asbm_{opt.problem_name}_{opt.sde_type}_{sde_arg}_nfe{opt.forward_nfe}{fm}"
    if opt.distill:
        name += f"_distill_{opt.distill_conv}_t{opt.distill_cond_t}"
    return name


def set():
    opt = build_parser().parse_args()

    # --------------- data ---------------
    if opt.problem_name == "cifar10":
        opt.data_dim = [3, 32, 32]
        opt.ldm = False
    elif opt.problem_name == "ffhq":
        opt.data_dim = [4, 32, 32]  # SD-VAE latents of 256x256 images
        opt.ldm = True
        if opt.dataset_path is None and not opt.eval_fid:
            raise ValueError("--dataset-path (pre-computed FFHQ latents) is required for training.")

    if opt.start_from_sm and opt.load is None:
        raise ValueError("--start-from-sm requires --load.")
    if opt.eval_fid and opt.load is None:
        raise ValueError("--eval-fid requires --load.")
    if opt.score_matching:
        if opt.sde_type not in ["vp", "vp_const"] or opt.flow_matching:
            raise ValueError("--score-matching supports the VP reference SDE only.")
        if opt.start_from_sm and opt.reinit_score_model:
            raise ValueError("--reinit-score-model is not used with --score-matching.")
    if opt.distill:
        if opt.problem_name != "cifar10":
            raise ValueError("--distill is implemented for CIFAR-10.")
        if opt.load is None:
            raise ValueError("--distill requires --load (teacher checkpoint, or distilled checkpoint with --eval-fid).")
        if opt.sde_type not in ["vp", "vp_const"] or opt.flow_matching:
            raise ValueError("--distill supports teachers with the VP reference SDE.")

    # --------------- seed ---------------
    random.seed(opt.seed)
    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = True

    opt.device = torch.device(f"cuda:{opt.gpu}")

    # --------------- experiment directory ---------------
    if not opt.eval_fid:
        if opt.name is None:
            opt.name = default_name(opt)
            if opt.load is not None:
                opt.name += "_resumed"
        opt.ckpt_path = os.path.join(opt.ckpt_root, opt.name)
        os.makedirs(opt.ckpt_path, exist_ok=True)
        args = {k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(opt).items()}
        with open(os.path.join(opt.ckpt_path, "args.json"), "w") as f:
            json.dump(args, f, indent=2)
        with open(os.path.join(opt.ckpt_path, "command.txt"), "w") as f:
            f.write("python " + " ".join(sys.argv))

    for k, v in vars(opt).items():
        print(util.green(k), ":", util.yellow(v))
    print()
    return opt
