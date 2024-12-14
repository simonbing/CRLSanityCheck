from absl import flags, app
import wandb

from crc.apps import TrainMethod, EvaluateMethod

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('out_dir',
                    '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Directory to save results to.')
flags.DEFINE_enum('method', 'multiview_iv',
                  ['multiview_iv', 'contrast_crl'],
                  'Representation learning method')
flags.DEFINE_enum('dataset', 'lt_camera_v1', ['lt_camera_v1', 'contrast_synthetic'], 'Dataset.')
flags.DEFINE_enum('task', 'lt_scm_2', ['synth_reprod', 'lt_scm_2'], 'Experimental task.')
flags.DEFINE_string('data_root',
                    '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/data/chamber_downloads',
                    'Root directory of data.')
flags.DEFINE_bool('overwrite', False, 'Whether to overwrite saved datasets.')
flags.DEFINE_string('run_name', None, 'Name of experimental run.')
flags.DEFINE_integer('lat_dim', 5, 'Latent dimension.')
flags.DEFINE_enum('encoder', 'conv', ['fc', 'conv'], 'Encoder type.')
flags.DEFINE_integer('bs', 512, 'Batch size.')
flags.DEFINE_integer('epochs', 10, 'Training epochs.')
flags.DEFINE_integer('val_step', 10, 'Validation frequency (in epochs).')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_list('metrics', 'mcc', ['mcc', 'shd'], 'Evaluation metrics.')

# Contrast CRL flags
flags.DEFINE_float('kappa', 0.1, 'Kappa.')
flags.DEFINE_float('eta', 0.0001, 'Eta.')
flags.DEFINE_float('mu', 0.00001, 'Mu')

# Multiview flags
flags.DEFINE_integer('n_envs', 1,
                     'Number of interventional environments for multiview data.')
flags.DEFINE_enum('selection', 'ground_truth', ['ground_truth'],
                  'Selection for estimating content indices.')
flags.DEFINE_float('tau', 1.0, 'Temperature parameter for multiview loss.')


def main(argv):
    WANDB = True

    wandb_config = dict(
        model=FLAGS.method,
        dataset=FLAGS.dataset,
        task=FLAGS.task,
        run_name=FLAGS.run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.bs,
        epochs=FLAGS.epochs,
        lat_dim=FLAGS.lat_dim
    )

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if WANDB else 'offline',  # don't log if debugging
        config=wandb_config
    )

    # Dict with additional flags that not each method uses
    match FLAGS.method:
        case 'contrast_crl':
            kwarg_dict = {
                'kappa': FLAGS.kappa,
                'eta': FLAGS.eta,
                'mu': FLAGS.mu
            }

    train_method = TrainMethod(method=FLAGS.method,
                               out_dir=FLAGS.out_dir,
                               seed=FLAGS.seed,
                               dataset=FLAGS.dataset,
                               task=FLAGS.task,
                               data_root=FLAGS.data_root,
                               overwrite=FLAGS.overwrite,
                               run_name=FLAGS.run_name,
                               d=FLAGS.lat_dim,
                               batch_size=FLAGS.bs,
                               epochs=FLAGS.epochs,
                               val_step=FLAGS.val_step,
                               lr=FLAGS.lr,
                               encoder=FLAGS.encoder,
                               **kwarg_dict)
    train_method.run()

    eval_method = EvaluateMethod(method=FLAGS.method,
                                 out_dir=FLAGS.out_dir,
                                 seed=FLAGS.seed,
                                 dataset=FLAGS.dataset,
                                 task=FLAGS.task,
                                 run_name = FLAGS.run_name,
                                 metrics=FLAGS.metrics,
                                 trained_method=train_method.method)
    eval_method.run()


if __name__ == '__main__':
    app.run(main)
