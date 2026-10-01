import os
from glob import glob
from pathlib import Path

import torch

import distill
import options
import util
from runner import Runner


def eval_fid(opt):
    from cleanfid import fid

    gen_dir = opt.gen_dir
    if gen_dir is None:
        ckpt = Path(opt.load)
        sampler = "1step" if opt.distill else f"{opt.sampler}_nfe{opt.sample_nfe}"
        gen_dir = os.path.join(ckpt.parent, f"samples_{ckpt.stem}_{sampler}_seed{opt.seed}")

    if len(glob(os.path.join(gen_dir, "*.png"))) >= opt.num_fid_samples:
        print(util.yellow(f"{opt.num_fid_samples} images already exist in {gen_dir}; skip generation."))
    else:
        print(util.magenta(f"Generating {opt.num_fid_samples} images into {gen_dir}"))
        if opt.distill:
            distill.generate(distill.load_generator(opt), opt, gen_dir, opt.num_fid_samples, opt.fid_bs)
        else:
            Runner(opt).generate(gen_dir, opt.num_fid_samples, opt.fid_bs, opt.sample_nfe, opt.sampler)

    if opt.problem_name == "cifar10":
        score = fid.compute_fid(gen_dir, dataset_name="cifar10", dataset_res=32,
                                dataset_split="train", mode="legacy_tensorflow")
    elif opt.problem_name == "ffhq":
        score = fid.compute_fid(gen_dir, dataset_name="ffhq", dataset_res=256,
                                dataset_split="trainval70k", mode="legacy_tensorflow")
    else:
        raise NotImplementedError
    setting = "one-step generator" if opt.distill else f"{opt.sampler}, NFE={opt.sample_nfe}"
    print(util.green(f"FID ({opt.problem_name}, {setting}): {score:.4f}"))
    return score


def main():
    print(util.yellow("======================================================="))
    print(util.yellow("     ASBM"))
    print(util.yellow("======================================================="))
    opt = options.set()

    with torch.cuda.device(opt.device):
        if opt.eval_fid:
            eval_fid(opt)
        elif opt.distill:
            distill.train_distill(Runner(opt))
        else:
            Runner(opt).train()


if __name__ == "__main__":
    main()
