# Adapted from SB-FBSDE (https://github.com/ghliu/SB-FBSDE).
import termcolor
import torch


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(runner, keys, fn):
    checkpoint = {k: getattr(runner, k).state_dict() for k in keys}
    torch.save(checkpoint, fn)
    print(green("checkpoint saved: {}".format(fn)))

def restore_checkpoint(runner, load_name):
    """Load every module stored in the checkpoint. Released checkpoints contain only the EMA networks
    (`ema_f`, `ema_b`); training checkpoints also contain the raw networks and optimizers."""
    print(green("#loading checkpoint {}...".format(load_name)))
    checkpoint = torch.load(load_name, map_location="cpu")
    for k, state_dict in checkpoint.items():
        getattr(runner, k).load_state_dict(state_dict)
    print(green("#loaded modules: {}".format(list(checkpoint.keys()))))
    return list(checkpoint.keys())
