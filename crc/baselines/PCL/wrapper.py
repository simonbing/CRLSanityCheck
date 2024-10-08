from causalchamber.datasets import Dataset as ChamberDataset

from crc.wrappers import TrainModel, EvalModel
from crc.baselines.PCL.pcl.train import train


class TrainPCL(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        # Get data
        pass


class EvalPCL(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        pass

    def get_encodings(self, dataset_test):
        pass
