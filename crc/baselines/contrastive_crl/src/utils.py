import os
import numbers

import torch
from PIL import Image
import numpy as np

from crc.baselines.contrastive_crl.src.data_generation import get_data_from_kwargs


def sanity_checks_kwargs(data_kwargs, model_kwargs, training_kwargs):
    if isinstance(data_kwargs['var_range_int'], numbers.Number):
        data_kwargs['var_range_int'] = [data_kwargs['var_range_int'], data_kwargs['var_range_int']]
    if isinstance(data_kwargs['var_range_obs'], numbers.Number):
        data_kwargs['var_range_obs'] = [data_kwargs['var_range_obs'], data_kwargs['var_range_obs']]
    if isinstance(data_kwargs['mean_range'], numbers.Number):
        data_kwargs['mean_range'] = [data_kwargs['mean_range'], data_kwargs['mean_range']]
    if 'device' in training_kwargs.keys():
        if training_kwargs['device'] == 'mps' and not torch.backends.mps.is_available():
            print('Device mps is not available defaulting to cpu!')
            training_kwargs['device'] = 'cpu'
        if training_kwargs['device'] == 'cuda' and not torch.cuda.is_available():
            print('Cuda is not available defaulting to cpu!')
            training_kwargs['device'] = 'cpu'

    model_kwargs['image'] = True if data_kwargs['mixing'] == 'image' else False
    if data_kwargs['mixing'] == 'image':
        data_kwargs['dim_x'] = 64 * 64 * 3
        if data_kwargs['d'] % 2 != 0:
            print('Only even d allowed for image dataset, rounding down')
            data_kwargs['d'] = (data_kwargs['d'] // 2) * 2
        if max(data_kwargs['var_range_obs'][1], data_kwargs['var_range_obs'][1]) > .3:
            print('Careful this variance is too large for image dataset')
        if data_kwargs['mean_range'][1] > .4:
            print('Careful this mean_shift is too large for image dataset')
        if training_kwargs['run_baseline']:
            print('Baseline not meaningful on image dataset, skipping this')
            training_kwargs['run_baseline'] = False
    if data_kwargs['mixing'] != 'image':
        if data_kwargs.get('constrain_to_image'):
            data_kwargs['constrain_to_image'] = False

    model_kwargs['input_dim'] = data_kwargs['dim_x']
    model_kwargs['latent_dim'] = data_kwargs['d']
    return data_kwargs, model_kwargs, training_kwargs


def generate_images(model, databag, directory, device='cpu', samples=5):
    model.eval()
    x = torch.tensor(databag.obs[:samples], device="cpu", dtype=torch.float)
    image_gt = databag.f(x)
    image_gt = image_gt.to(device)
    z = model.get_latents(image_gt)
    images = model.decoder(z).detach().cpu()
    save_images(image_gt, directory, 'true')
    save_images(images, directory, 'fake')


def save_images(images, dir, filename):
    generated_image_np = 1 - images.detach().cpu().numpy()
    generated_image_np = (generated_image_np * 255).astype(np.uint8)
    generated_image_np = np.transpose(generated_image_np, (0, 2, 3, 1))
    for i in range(generated_image_np.shape[0]):
        generated_image_pil = Image.fromarray(generated_image_np[i])

        generated_image_pil.save(os.path.join(dir, '{}_{}.png'.format(filename, i)))


def get_chamber_data(dataset, seed, batch_size):
    # For sanity checking contrastive CRL code
    if dataset == 'contrast_synth':
        mixing = 'mlp'  # TODO: this can also be 'image', make this an argument
        data_kwargs = {
            'mixing': mixing,
            'd': 5,
            'k': 2,
            'n': 10000,
            'seed': seed,
            'dim_x': 20,
            'hidden_dim': 512,
            'hidden_layers': 3,
            'var_range_obs': (1., 2.),
            'var_range_int': (1., 2.),
            'mean_range': (1., 2.)
        }  # TODO get these kwargs
        databag = get_data_from_kwargs(data_kwargs)  # databags is the term used in original code

        dataloader_obs, dataloader_int = databag.get_dataloaders(
            batch_size=batch_size, train=True)
        dataloader_obs_val, dataloader_int_val = databag.get_dataloaders(
            batch_size=batch_size, train=True)

        return dataloader_obs, dataloader_int, dataloader_obs_val, dataloader_int_val
