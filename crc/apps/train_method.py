import os

from absl import app, flags
import torch
import wandb

from crc.methods import ContrastCRL, MultiviewIv


class TrainMethod(object):
    def __init__(self, method, out_dir, run_name, **kwargs):
        self.method_name = method

        self.method = self._get_method(method=method)(**kwargs)

        self.model_dir = os.path.join(out_dir, kwargs['dataset'], kwargs['task'],
                                      self.method_name)
        self.train_dir = os.path.join(self.model_dir, run_name,
                                      f"seed_{kwargs['seed']}", 'train')
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

    @staticmethod
    def _get_method(method):
        match method:
            case 'contrast_crl':
                return ContrastCRL
            case 'multiview_iv':
                return MultiviewIv

            case _:
                AssertionError(f'Undefined method {method}!')

    def run(self):
        # Check if trained model already exists, skip training if so
        if os.path.exists(os.path.join(self.train_dir, 'best_model.pt')):
            print('Trained model found, skipping training!')
            return

        # Training
        best_model = self.method.train()
        print('Training finished!')

        # Save model
        torch.save(best_model, os.path.join(self.train_dir, 'best_model.pt'))


if __name__ == '__main__':
    SEED = 0
    OUT_DIR = '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results'
    METHOD = 'multiview_iv'
    DATASET = 'lt_camera_v1'
    TASK = 'lt_scm_2'
    DATA_ROOT = '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/data/chamber_downloads'
    RUN_NAME = 'multiview_test_0'
    D = 5
    ENCODER = 'conv'
    BS = 512
    EPOCHS = 10
    LR = 0.0001
    # Multiview Args
    N_ENVS = 1
    SELECTION= 'ground_truth'
    TAU = 1.0
    kwarg_dict = {'n_envs': N_ENVS,
                  'selection': SELECTION,
                  'tau': TAU}

    WANDB = False

    wandb_config = dict(
        model=METHOD,
        dataset=DATASET,
        task=TASK,
        run_name=RUN_NAME,
        seed=SEED,
        batch_size=BS,
        epochs=EPOCHS,
        lat_dim=D
    )

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if WANDB else 'offline',
        # don't log if debugging
        config=wandb_config
    )

    method = TrainMethod(method=METHOD, out_dir=OUT_DIR, seed=SEED, dataset=DATASET, task=TASK,
                         data_root=DATA_ROOT, run_name=RUN_NAME, d=D, batch_size=BS, epochs=EPOCHS,
                         lr=LR, encoder=ENCODER, **kwarg_dict)

    app.run(method.run())
