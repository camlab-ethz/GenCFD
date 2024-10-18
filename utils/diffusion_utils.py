from argparse import ArgumentParser
import torch
import diffusion.diffusion as dfn_lib

def get_diffusion_scheme(args: ArgumentParser, device: torch.device = None) -> dfn_lib.Diffusion:
    """Create the diffusion scheme"""

    try:
        diffusion_scheme_fn = getattr(dfn_lib.Diffusion, args.diffusion_scheme)
    except AttributeError:
        raise ValueError(f"Invalid diffusion scheme: {args.diffusion_scheme}")
    
    try:
        sigma_fn = getattr(dfn_lib, args.sigma)
    except AttributeError:
        raise ValueError(f"Invalid sigma function: {args.sigma}")

    diffusion_scheme = diffusion_scheme_fn(
        sigma=sigma_fn(device=device),
        data_std=args.sigma_data,
    )

    return diffusion_scheme

def get_noise_sampling(args: ArgumentParser, device: torch.device = None) -> dfn_lib.NoiseLevelSampling:
    """Create a noise sampler"""

    diffusion_scheme = get_diffusion_scheme(args, device)
    try:
        noise_sampling_fn = getattr(dfn_lib, args.noise_sampling)
    except AttributeError:
        raise ValueError(f"Invalid noise sampling scheme: {args.noise_sampling}")
    
    noise_sampling = noise_sampling_fn(
        diffusion_scheme, 
        clip_min=1e-4, 
        uniform_grid=True, 
        device=device
    )

    return noise_sampling


def get_noise_weighting(args: ArgumentParser, device: torch.device = None) -> dfn_lib.NoiseLossWeighting:
    """Create a noise weighting scheme"""

    try:
        noise_weighting_fn = getattr(dfn_lib, args.noise_weighting)
    except AttributeError:
        raise ValueError(f"Invalid noise weighting scheme: {args.noise_weighting}")
    
    noise_weighting = noise_weighting_fn(
        data_std=args.sigma_data,
        device=device
    )

    return noise_weighting
