from absl import flags, app

class EvalApplication(object):
    def __init__(self):
        pass

    def run(self):
        # Load test data (all ground truth info should be here!)

        # Load trained model

        # Evaluate all methods
        pass

def main(argv):
    application = EvalApplication()
    application.run()


if __name__ =='__main__':
    app.run(main)
