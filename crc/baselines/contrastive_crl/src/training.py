import copy

from tqdm import tqdm
import torch
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from crc.baselines.contrastive_crl.src.evaluation import LossCollection, get_R2_values


def get_NOTEARS_loss(A):
    return torch.trace(torch.matrix_exp(A * A) - A.size(0))


def train_model(model, dl_obs, dl_int, dl_obs_test, dl_int_test, training_kwargs, z_gt=None, x_val=None, verbose=False):
    device = training_kwargs.get("device", 'cpu')
    model = model.to(device)
    best_model = copy.deepcopy(model)

    mse = torch.nn.MSELoss()
    mse = torch.nn.HuberLoss(delta=1., reduction='sum')
    ce = torch.nn.CrossEntropyLoss()
    loss_tracker = LossCollection()
    val_loss = np.inf

    epochs = training_kwargs.get("epochs", 10)
    mu = training_kwargs.get('mu', 0.0)
    eta = training_kwargs.get('eta', 0.0)
    kappa = training_kwargs.get('kappa', 0.0)
    lr_nonparametric = training_kwargs.get('lr_nonparametric', .1)
    lr_parametric = training_kwargs.get('lr_parametric', lr_nonparametric)
    contrastive = False if training_kwargs.get("type") in ['vae', 'vae_vanilla', 'vae_vanilla2', 'vae_contrastive'] else True
    optimizer_name = training_kwargs.get("optimizer", "sgd").lower()

    non_parametric_params = []
    if hasattr(model, "embedding"):
        non_parametric_params += list(model.embedding.parameters())
    if hasattr(model, "encoder"):
        non_parametric_params += list(model.encoder.parameters())
    if hasattr(model, "decoder"):
        non_parametric_params += list(model.decoder.parameters())

    if optimizer_name == 'sgd':
        optim = torch.optim.SGD([
                {'params': model.parametric_part.parameters(), 'lr': lr_parametric},
                {'params': non_parametric_params, 'lr': lr_nonparametric}
            ], weight_decay=training_kwargs.get('weight_decay', 0.0))
    elif optimizer_name == 'adam':
        optim = torch.optim.Adam([
                {'params': model.parametric_part.parameters(), 'lr': lr_parametric},
                {'params': non_parametric_params, 'lr': lr_nonparametric}
            ], weight_decay=training_kwargs.get('weight_decay', 0.0))
    else:
        raise NotImplementedError("Only Adam and SGD supported at the moment")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.3, patience=3)

    train_loss_history = []
    val_loss_history = []
    r2_history = []

    for i in tqdm(range(epochs)):
    # for i in range(epochs):
        model.train()
        for step, data in enumerate(zip(dl_obs, dl_int)):
            x_obs, x_int, t_int = data[0], data[1][0], data[1][1]
            x_obs, x_int, t_int = x_obs.to(device), x_int.to(device), t_int.to(device)
            if contrastive:
                logits_int = model(x_int, t_int)
                logits_obs, embedding = model(x_obs, t_int, True)
                method_specific_loss = kappa * torch.sum(torch.mean(embedding, dim=0) ** 2)

            else:
                x_int_hat, mean_int, logvar_int, logits_int = model(x_int, t_int, True)
                x_obs_hat, mean_obs, logvar_obs, logits_obs = model(x_obs, t_int, True)

                rec_loss = mse(x_obs, x_obs_hat) / x_obs.size(0)
                # learn only to reconstruct observational distribution
                # rec_loss = mse(x_obs, x_obs_hat) / x_obs.size(0)
                kl_divergence = - 0.5 * torch.mean(1 + logvar_obs - mean_obs.pow(2) - logvar_obs.exp())
                if not model.match_observation_dist_only:
                    rec_loss += mse(x_int, x_int_hat) / x_int.size(0)
                    kl_divergence += - 0.5 * torch.mean(1 + logvar_int - mean_int.pow(2) - logvar_int.exp())
                method_specific_loss = rec_loss + kl_divergence

            classifier_loss = ce(logits_obs, torch.zeros(x_obs.size(0), dtype=torch.long, device=device)) + \
                              ce(logits_int, torch.ones(x_int.size(0), dtype=torch.long, device=device))
            accuracy = (torch.sum(torch.argmax(logits_obs, dim=1) == 0) +
                        torch.sum(torch.argmax(logits_int, dim=1) == 1)) / (2 * x_int.size(0))
            if device == 'mps':
                reg_loss = eta * torch.sum(torch.abs(model.parametric_part.A))
            else:
                reg_loss = eta * torch.sum(torch.abs(model.parametric_part.A)) + mu * get_NOTEARS_loss(model.parametric_part.A)

            loss = method_specific_loss + classifier_loss + reg_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_tracker.add_loss(
                {'method_loss': method_specific_loss.item(), 'CE-loss': classifier_loss.item(),
                 'A-reg loss': reg_loss.item(), 'accuracy': accuracy.item()}, x_obs.size(0))
        if verbose:
            print("Finished epoch {}, printing test and validation loss".format(i + 1))
            loss_tracker.print_mean_loss()
        if getattr(model, 'vanilla', False):
            train_loss_history.append(loss_tracker.get_mean_loss()['method_loss'])
        else:
            train_loss_history.append(loss_tracker.get_mean_loss()['CE-loss'])
        loss_tracker.reset()
        for step, data in enumerate(zip(dl_obs_test, dl_int_test)):
            x_obs, x_int, t_int = data[0], data[1][0], data[1][1]
            x_obs, x_int, t_int = x_obs.to(device), x_int.to(device), t_int.to(device)
            if contrastive:
                logits_int = model(x_int, t_int)
                logits_obs, embedding = model(x_obs, t_int, True)
                method_specific_loss = eta * torch.sum(torch.mean(embedding, dim=0) ** 2)
            else:
                x_int_hat, mean_int, _, logits_int = model(x_int, t_int, False)
                x_obs_hat, mean_obs, logvar_obs, logits_obs = model(x_obs, t_int, False)

                rec_loss = mse(x_obs, x_obs_hat) / x_obs.size(0)
                if not model.match_observation_dist_only:
                    rec_loss += mse(x_int, x_int_hat) / x_int.size(0)

                kl_divergence = - 0.5 * torch.mean(1 + logvar_obs - mean_obs.pow(2) - logvar_obs.exp())
                method_specific_loss = rec_loss + kl_divergence

            classifier_loss = ce(logits_obs, torch.zeros(x_obs.size(0), dtype=torch.long, device=device)) + \
                                  ce(logits_int, torch.ones(x_int.size(0), dtype=torch.long, device=device))
            accuracy = (torch.sum(torch.argmax(logits_obs, dim=1) == 0) + torch.sum(
            torch.argmax(logits_int, dim=1) == 1)) / (2 * x_int.size(0))
            loss_tracker.add_loss(
                {'method_loss': method_specific_loss.item(), 'CE-loss': classifier_loss.item(),
                 'A-reg loss': reg_loss.item(), 'accuracy': accuracy.item()}, x_obs.size(0))
        ce_loss = loss_tracker.get_mean_loss()['CE-loss']
        # vanilla models do not provide classification so we use their elbo for model selection
        if getattr(model, 'vanilla', False):
            ce_loss = loss_tracker.get_mean_loss()['method_loss']
        if ce_loss < val_loss:
            best_model = copy.deepcopy(model)
            val_loss = ce_loss
        if verbose:
            loss_tracker.print_mean_loss()
        val_loss_history.append(ce_loss)

        # scheduler.step()
        scheduler.step(ce_loss)

        loss_tracker.reset()
        if z_gt is not None:
            z_gt_tensor = torch.tensor(z_gt, device=device, dtype=torch.float)
            z_pred = model.get_z(torch.tensor(x_val, device=device, dtype=torch.float))
            r2_history.append(torch.mean(get_R2_values(z_gt_tensor, z_pred)).item())
        else:
            r2_history.append(0)
    return best_model, model, val_loss, [train_loss_history, val_loss_history, r2_history]

