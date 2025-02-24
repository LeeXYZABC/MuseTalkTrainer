# from Wav2Lip https://github.com/Rudrabha/Wav2Lip

import os
import torch
import torch.nn as nn


def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def _load(checkpoint_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_weights(path, model, use_cuda=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda=use_cuda)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    return model



def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True, use_cuda=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda=use_cuda)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model, optimizer, global_step, global_epoch


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = os.path.join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


