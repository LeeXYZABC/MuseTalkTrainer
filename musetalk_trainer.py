import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as data_utils

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding
from discriminator import VQGANDiscriminator, GANLoss

import utils
import musetalk_loss
from musetalk_datasets import TrainingDataset
from msglogger import msglogger, msglogger_init


base_dir = os.path.split(os.path.realpath(__file__))[0]
log_path = os.path.join(base_dir, "./log")
if not os.path.exists(log_path):
    os.makedirs(log_path)
msglogger_init(log_path, "trainer")


device = "cuda"
eps = 1e-8
weight_decay = 0.0
betas= (0.9, 0.999)
initial_learning_rate=1e-4



class Loss:
    def __init__(self, device="cpu"):
        self.device = device
        self.lpips = musetalk_loss.LPIPSLoss().to(device)
        self.criterion = nn.MSELoss().to(device)
        self.l1 = nn.L1Loss(reduction='mean').to(device)
        self.cri_gan = GANLoss(gan_type="hinge")

    def toRGBTensor(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def calc_loss(self, latents_pred, latents_gt, image_gt, image_pred, fake_g_pred):
        latents_loss = self.l1(latents_pred, latents_gt)
        image_loss = self.l1(image_pred, image_gt)
        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
        lpips_loss = self.lpips(self.toRGBTensor(image_pred), self.toRGBTensor(image_gt))
        loss = latents_loss + image_loss +  0.01 * l_g_gan + 0.01 * lpips_loss
        return loss, latents_loss, image_loss, l_g_gan, lpips_loss

 
def run_epoch(
    data_loader,
    unet, optimizer, pe, vae, timesteps,
    net_d, optimizer_d,
    checkpoint_dir,
    global_step, curr_epoch,
    scaler, scaler_d,
    loss_module
):
    total_loss = 0.0
    l_loss = 0.0
    i_loss = 0.0
    g_loss = 0.0
    d_real = 0.0
    d_fake = 0.0
    lpips_l = 0.0
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        unet.model.train()
        net_d.train()
        vae.vae.eval()
        optimizer.zero_grad()

        latents_gt = batch["latents_gt"]
        latents_input = batch["latents_input"]
        whisper_input = batch["whisper_input"]
        image_gt = batch["image_gt"]

        latents_gt = latents_gt.to(device, dtype=unet.model.dtype)
        latents_input = latents_input.to(device, dtype=unet.model.dtype)
        whisper_input = whisper_input.to(device, dtype=unet.model.dtype)
        image_gt = image_gt.to(device, dtype=unet.model.dtype)
        
        with autocast(dtype=torch.float16): 
            whisper_input = pe(whisper_input)
            latents_pred = unet.model(latents_input, timesteps, encoder_hidden_states=whisper_input).sample
            image_pred = vae.decode_only(latents_pred)
            fake_g_pred = net_d(image_pred)
            loss, latents_loss, image_loss, l_g_gan, lpips_loss = \
                    loss_module.calc_loss(latents_pred.float(), latents_gt.float(), image_gt.float(), image_pred.float(), fake_g_pred.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer_d.zero_grad()
        # real
        with autocast(dtype=torch.float16):    
            real_d_pred = net_d(image_gt)
            l_d_real = loss_module.cri_gan(real_d_pred, True, is_disc=True)
        scaler_d.scale(l_d_real).backward()
        # fake
        with autocast(dtype=torch.float16):    
            fake_d_pred = net_d(image_pred.detach())
            l_d_fake = loss_module.cri_gan(fake_d_pred, False, is_disc=True)
        scaler_d.scale(l_d_fake).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        total_loss += loss.item()/100.
        l_loss += latents_loss.item()/100.
        i_loss += image_loss.item()/100.
        g_loss += l_g_gan.item()/100.
        d_real += l_d_real.item()/100.
        d_fake += l_d_fake.item()/100.
        lpips_l += lpips_loss.item()/100.
        
        if i % 100 == 0:
            msglogger.info("step:{}, ep:{}, loss:{}, lanloss:{}, imgloss:{}, gloss:{}, dreal:{}, dfake:{}, lpips: {}".format(
                        global_step, curr_epoch,
                        round(total_loss, 3),
                        round(l_loss, 3),
                        round(i_loss, 3),
                        round(g_loss, 3),
                        round(d_real, 3),
                        round(d_fake, 3),
                        round(lpips_l, 3)
                    )
                 )
            total_loss = 0.0
            l_loss = 0.0
            i_loss = 0.0
            d_real = 0.0
            d_fake = 0.0
            g_loss = 0.0
            lpips_l = 0.0

        if global_step % 10000 == 0:
            utils.save_checkpoint(unet.model, optimizer, global_step,
                                  checkpoint_dir, curr_epoch,
                                  prefix="unet_")
            utils.save_checkpoint(net_d, optimizer_d, global_step,
                                  checkpoint_dir, curr_epoch,
                                  prefix="disc_")
        
        global_step += 1

    return global_step

    
        
def run_training(train_data_loader,
                 unet, optimizer, pe, vae, timesteps,
                 net_d, optimizer_d,
                 checkpoint_dir,
                 global_step, start_epoch, total_epoches):
    loss_module = Loss(device)

    scaler = GradScaler()
    scaler_d = GradScaler()

    for curr_epoch in range(start_epoch, total_epoches):
        global_step = run_epoch(
            train_data_loader,
            unet, optimizer, pe, vae, timesteps,
            net_d, optimizer_d,
            checkpoint_dir,
            global_step, curr_epoch,
            scaler, scaler_d,
            loss_module)

    
def run_main():
    global_step = 0
    global_epoch = 0
    batch_size = 10
    total_epoches = 36

    hdtf_samples_dir = "/opt/datasets/hdtf/samples"
    
    checkpoint_dir = "./checkpoints"
    unet_checkpoint_path = None # "checkpoints/unet_checkpoint_step000250000.pth"
    disc_checkpoint_path = None # "checkpoints/disc_checkpoint_step000250000.pth"
    reset_optimizer = False

    vae = VAE(model_path = "./models/sd-vae-ft-mse/", device=device)
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path=None, device=device)
    pe = PositionalEncoding(d_model=384).to(device)
    timesteps = torch.tensor([0], device=device)

    net_d = VQGANDiscriminator(nc=3, ndf=64, n_layers=3).to(device)
    net_d = utils.add_sn(net_d)
    
    # optim.AdamW
    # optimizer = optim.AdamW(
    #     [p for p in unet.model.parameters() if p.requires_grad],
    #     lr=initial_learning_rate,
    #     betas=betas,
    #     weight_decay=weight_decay
    #     # eps=eps
    # )
    # optimizer_d = optim.AdamW(
    #     [p for p in net_d.parameters() if p.requires_grad],
    #     lr=initial_learning_rate, 
    #     betas=betas,
    #     weight_decay=weight_decay
    # )

    # optim.Adamax
    optimizer = optim.Adamax(
        [p for p in unet.model.parameters() if p.requires_grad],
        lr=initial_learning_rate, 
        betas=betas
    )
    optimizer_d = optim.Adamax(
        [p for p in net_d.parameters() if p.requires_grad],
        lr=initial_learning_rate, 
        betas=betas
    )

    torch.cuda.empty_cache()

    if unet_checkpoint_path:
        unet.model, optimizer, global_step, global_epoch = utils.load_checkpoint(
                unet_checkpoint_path, unet.model, optimizer,
                reset_optimizer=reset_optimizer,
                overwrite_global_states=True,
                use_cuda=("cuda" in device)
            )

    if disc_checkpoint_path:
        net_d, optimizer_d, global_step, global_epoch = utils.load_checkpoint(
                disc_checkpoint_path, net_d, optimizer_d,
                reset_optimizer=reset_optimizer,
                overwrite_global_states=True,
                use_cuda=("cuda" in device)
            )
        
    global_step = global_step + 1

    train_dataset = TrainingDataset(hdtf_dir=hdtf_samples_dir)

    start_epoch = global_step // max(len(train_dataset)//batch_size, 1)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    run_training(train_data_loader,
                 unet, optimizer, pe, vae, timesteps,
                 net_d, optimizer_d, 
                 checkpoint_dir,
                 global_step, start_epoch, total_epoches)



if __name__ == "__main__":
    run_main()