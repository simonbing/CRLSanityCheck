import sys
import time

from absl import flags, app
import wandb

from crc.apps import OODEstimatorApplication


FLAGS = flags.FLAGS

flags.DEFINE_enum('estimation_model', None, ['ols', 'lasso', 'mlp', 'cmvae',
                                             'contrast_crl'], 'Estimator to train.')
flags.DEFINE_enum('task', None, ['lt_1'], 'Prediction task.')
flags.DEFINE_string('output_root', '/Users/Simon/Documents/PhD/Projects/'
                                   'CausalRepresentationChambers/results',
                    'Root directory where output is saved.')
flags.DEFINE_string('data_root', '/Users/Simon/Documents/PhD/Projects/'
                                 'CausalRepresentationChambers/data/chamber_downloads',
                    'Root directory where data is saved.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')


def main(argv):
    if FLAGS.run_name is None:
        run_name = str(int(time.time()))
    else:
        run_name = FLAGS.run_name

    wandb_config = dict(
        estimation_model=FLAGS.estimation_model,
        task=FLAGS.task,
        run_name=run_name,
        seed=FLAGS.seed
    )

    gettrace = getattr(sys, 'gettrace', None)

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if not gettrace() else 'offline',
        # don't log if debugging
        config=wandb_config
    )

    ood_app = OODEstimatorApplication(seed=FLAGS.seed,
                                      estimation_model=FLAGS.estimation_model,
                                      task=FLAGS.task,
                                      data_root=FLAGS.data_root,
                                      epochs=FLAGS.epochs,
                                      learning_rate=FLAGS.learning_rate,
                                      batch_size=FLAGS.batch_size)

    ood_app.run()


if __name__ == '__main__':
    app.run(main)
