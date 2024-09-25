from sklearn.linear_model import LinearRegression

from crc.ood_estimation.base_estimator import OODEstimator


class OLSOODEstimator(OODEstimator):
    def __init__(self, seed, task):
        super().__init__(seed, task)

        self.X_train, self.y_train = self._get_train_data()

        self.model = LinearRegression()

    def _get_train_data(self):
        if self.task == 'lt_1':
            # Return training data
            X_train = None
            y_train = None
        return X_train, y_train

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_ood):
        y_hat = self.model.predict(X_ood)

        return y_hat
