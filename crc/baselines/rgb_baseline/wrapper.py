from crc.wrappers import TrainModel, EvalModel


class TrainRGBBaseline(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        # Just get model, dont need to train anything
        # save model for eval later (and ood)
        # Also probably need to save datasets here
        pass


class EvalRGBBaseline(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        pass

    def get_encodings(self, dataset_test):
        # Iterate over dataset,
        pass
