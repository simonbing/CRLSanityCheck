from absl import flags, app

from crc.apps import EvalApplication

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Root directory for experiments.')
flags.DEFINE_enum('model', None, ['cmvae', 'contrast_crl'], 'Model to evaluate.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1', 'contrast_synth'], 'Dataset for training.')
flags.DEFINE_string('experiment', None, 'Experiment for training.')
flags.DEFINE_string('run_name', None, 'Run name where trained model is saved.')
flags.DEFINE_list('metrics', None, 'Evaluation metrics to calculate.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(argv):
    application = EvalApplication(seed=FLAGS.seed,
                                  model=FLAGS.model,
                                  root_dir=FLAGS.root_dir,
                                  dataset=FLAGS.dataset,
                                  experiment=FLAGS.experiment,
                                  run_name=FLAGS.run_name,
                                  metrics=FLAGS.metrics)
    application.run()


if __name__ == '__main__':
    app.run(main)
