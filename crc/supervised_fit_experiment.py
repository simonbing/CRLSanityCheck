import os
import sys

from absl import app, flags
from causalchamber.datasets import Dataset as ChamberData
import numpy as np
from skimage import io
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
import wandb

from crc.utils import get_device
from crc.shared.architectures import FCEncoder

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', './data/chamber_downloads',
                    'Root directory where data is saved.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')


class SupervisedChambersDataset(Dataset):
    def __init__(self, dataset, dataroot, exp):
        super().__init__()

        chamber_data = ChamberData(name=dataset,
                                   root=dataroot,
                                   download=True)

        self.data_df = chamber_data.get_experiment(name=exp).as_pandas_dataframe()

        latents = self.data_df[['red', 'green', 'blue', 'pol_1', 'pol_2']].to_numpy()
        self.latents = (latents - np.mean(latents, axis=0, keepdims=True)) / np.std(latents, axis=0, keepdims=True)

        self.base_img_path = os.path.join(dataroot, dataset, exp, 'images_64')

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        img = io.imread(os.path.join(self.base_img_path,
                                     self.data_df['image_file'].iloc[item]))
        img = img / 255.0
        img = img * 2.0 - 1.0

        latents = self.latents[item]

        return torch.as_tensor(img, dtype=torch.float32).flatten(), \
            torch.as_tensor(latents, dtype=torch.float32)


class SupervisedChambersModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        d = 5

        self.encoder = FCEncoder(in_dim=3*64*64,
                                 latent_dim=d,
                                 hidden_dims=[64, 256, 256, 64])

    def forward(self, x):
        y_hat = self.encoder(x)

        return y_hat

def main(argv):
    ############### wandb section ###############
    # Can be ignored if not using wandb for experiment tracking
    wandb_config = dict(
        model='supervised'
    )

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace() or None in [FLAGS.wandb_project, FLAGS.wandb_username]:
        print('Not using wandb for logging! This could be due to missing project and username flags!')
        wandb_mode = 'offline'
    else:
        wandb_mode = 'online'

    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_username,
        mode=wandb_mode,
        config=wandb_config
    )
    ##############################################

    # Get datasets
    train_dataset = Subset(SupervisedChambersDataset(
        dataset='lt_crl_benchmark_v1',
        dataroot=FLAGS.data_root,
        exp='citris_1'),
        np.arange(1000, 6000))
    test_dataset = Subset(SupervisedChambersDataset(
        dataset='lt_crl_benchmark_v1',
        dataroot=FLAGS.data_root,
        exp='citris_1'),
        np.arange(1000))

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = get_device()

    model = SupervisedChambersModel()
    model = model.to(device=device)

    optimizer = torch.optim.Adam(params=model.parameters())

    mse_loss = torch.nn.MSELoss()

    # Train model
    for epoch in tqdm(range(FLAGS.epochs)):
        model.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            # Zero gradients
            optimizer.zero_grad()

            y_hat = model(x)
            loss = mse_loss(y_hat, y)

            wandb.log({'loss': loss})

            loss.backward()
            optimizer.step()

    print('Training finished!')

    # Do eval
    model.eval()
    y_test_list = []
    y_hat_test_list = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = mse_loss(y_hat, y)

            wandb.log({'test loss': loss})

            y_test_list.append(y.detach().cpu().numpy())
            y_hat_test_list.append(y_hat.detach().cpu().numpy())

    y_test = np.concatenate(y_test_list, axis=0)
    y_hat_test = np.concatenate(y_hat_test_list, axis=0)

    # Get score, print results
    print(r2_score(y_test, y_hat_test))


if __name__ == '__main__':
    app.run(main)
