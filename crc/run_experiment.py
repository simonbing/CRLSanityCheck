import sys
import time

from absl import flags, app
import wandb

from crc.apps import TrainApplication, EvalApplication

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', None, ['pcl', 'cmvae', 'contrast_crl'], 'Model to train.')
flags.DEFINE_string('output_root', '/Users/Simon/Documents/PhD/Projects/'
                                   'CausalRepresentationChambers/results',
                    'Root directory where output is saved.')
flags.DEFINE_string('data_root', '/Users/Simon/Documents/PhD/Projects/'
                                 'CausalRepresentationChambers/data/chamber_downloads',
                    'Root directory where data is saved.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1', 'lt_camera_walks_v1',
                                    'contrast_synth', 'contrast_img', 'synth_pcl',
                                    'lt_camera_walks_v1_synth_mix'], 'Dataset for training.')
flags.DEFINE_bool('image_data', False, 'Indicates if input is image data.')
flags.DEFINE_string('task', None, 'Experimental task for training.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')
flags.DEFINE_bool('overwrite_data', False, 'Overwrite existing saved data.')

# Shared hyperparameters
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')
flags.DEFINE_integer('lat_dim', 5, 'Latent dimension.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')

flags.DEFINE_list('metrics', None, 'Evaluation metrics to calculate.')


def main(argv):
    if FLAGS.run_name is None:
        run_name = str(int(time.time()))
    else:
        run_name = FLAGS.run_name

    wandb_config = dict(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        task=FLAGS.task,
        run_name=run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        lat_dim=FLAGS.lat_dim
    )

    gettrace = getattr(sys, 'gettrace', None)

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if not gettrace() else 'offline',  # don't log if debugging
        config=wandb_config
    )

    train_app = TrainApplication(model=FLAGS.model,
                                 output_root=FLAGS.output_root,
                                 data_root=FLAGS.data_root,
                                 dataset=FLAGS.dataset,
                                 image_data=FLAGS.image_data,
                                 task=FLAGS.task,
                                 run_name=run_name,
                                 overwrite_data=FLAGS.overwrite_data,
                                 seed=FLAGS.seed,
                                 batch_size=FLAGS.batch_size,
                                 epochs=FLAGS.epochs,
                                 lat_dim=FLAGS.lat_dim)

    train_app.run()

    eval_app = EvalApplication(seed=FLAGS.seed,
                               model=FLAGS.model,
                               root_dir=FLAGS.output_root,
                               dataset=FLAGS.dataset,
                               image_data=FLAGS.image_data,
                               task=FLAGS.task,
                               run_name=FLAGS.run_name,
                               metrics=FLAGS.metrics)

    eval_app.run()


if __name__ == '__main__':
    app.run(main)
