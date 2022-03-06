import os
import sys

paths = ['../guided-diffusion/']
for path in paths:
    if os.path.abspath(path) not in sys.path:
        sys.path.insert(0, os.path.abspath(path))
        print(os.path.abspath(path))

import argparse
from importlib import reload
import numpy as np
import functools
import datetime

import torch
import torchvision
import torchvision.transforms as transforms

from glide_text2im.model_creation import model_and_diffusion_defaults, create_model
from glide_text2im.gaussian_diffusion import get_named_beta_schedule

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.script_util import add_dict_to_argparser, args_to_dict
from guided_diffusion.respace import space_timesteps, SpacedDiffusion
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion import dist_util, logger

from data import get_data

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    '''
    Copied from `https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/script_util.py`.
    Some beta_scheduler in glide-text2im is not supported in guided-diffusion.
    '''
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    
    learn_sigma,
    sigma_small,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    **kwargs
):
    '''
    https://github.com/openai/glide-text2im/blob/9cc8e563851bd38f5ddb3e305127192cb0f02f5c/glide_text2im/model_creation.py#L54
    '''
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def load_data(args, tokenizer):
    
    # TODO: check preprocessing correct or not
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.image_size),
        # transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    valid_transform = transforms.Compose([
        transforms.CenterCrop(args.image_size),
        # transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    def tokenize(text):
        tokens = tokenizer.encode(text)
        tokens, mask = tokenizer.padded_tokens_and_mask(
            tokens, args.text_ctx
        )
        cond = {'tokens': tokens, 'mask': mask}
        return cond

    data = get_data(args, (train_transform, valid_transform), 'glide', tokenize=tokenize)
    data.setup()
    data = iter(data.train_dataloader())
    return data

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        # train args
        log_dir="/home/andrewbai/glide_logs",
        image_size=256,
        # noised=True,
        iterations=150000,
        schedule_sampler="uniform",
        lr=1e-4,
        anneal_lr=False,
        lr_anneal_steps=0,
        weight_decay=0.0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        # eval_interval=5,
        save_interval=10000,
        resume_checkpoint="",
        learn_sigma=True, 
        sigma_small=False, 
        use_kl=False, 
        predict_xstart=False, 
        rescale_timesteps=False, 
        rescale_learned_sigmas=False,
        fp16_scale_growth=1e-3,
        # data args
        data_dir="../data",
        train_filename="../data/train.txt",
        valid_filename="../data/validation.txt",
        img_key="images",
        caption_key="captions",
        csv_separator=" ",
        permute=False,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    args = create_argparser().parse_args()
    
    model, diffusion = create_model_and_diffusion(**args_to_dict(args))

    data = load_data(args, model.tokenizer)
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    dist_util.setup_dist()
    timestamp = datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")
    logger.configure(dir=f'{args.log_dir}/{timestamp}')

    model = model.to(dist_util.dev())

    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )

    train_loop.run_loop()
    
if __name__ == '__main__':
    main()
