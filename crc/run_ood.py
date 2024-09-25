import sys
import time

from absl import flags, app
import wandb

from crc.apps import OODEstimatorApplication


FLAGS = flags.FLAGS


def main(argv):
    if FLAGS.run_name is None:
        run_name = str(int(time.time()))
    else:
        run_name = FLAGS.run_name

    ood_app = OODEstimatorApplication()

    ood_app.run()


if __name__ == '__main__':
    app.run(main)
