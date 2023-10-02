import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from implementations.trees.regression_tree import RegressionTree

np.random.seed(42)


class RandomRegressionForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_of_trees, bootstrap_fraction, features_fraction, max_depth):
        self.n_of_trees = n_of_trees
        self.bootstrap_fraction = bootstrap_fraction
        self.features_fraction = features_fraction
        self.max_depth = max_depth
        self._trees: list[RegressionTree] = []

    def fit(self, X, y):
        for _ in range(self.n_of_trees):
            X_frac, y_frac = self._bootstrap(X, y)
            X_frac = self._features_delete(X_frac)
            tree = RegressionTree(self.max_depth)
            tree.fit(X_frac, y_frac)
            self._trees.append(tree)

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self._trees])
        return np.sum(preds, axis=0) / self.n_of_trees

    def _bootstrap(self, X, y):
        to_return_n = self.bootstrap_fraction * len(X)
        to_return_n = int(to_return_n)
        return_indices = np.random.randint(low=0, high=len(X), size=(to_return_n, ))
        return X[return_indices], y[return_indices]

    def _features_delete(self, X):
        """ This somewhat mocks how features are selected in forests.
        The standard implementation would need changes in the decision tree class which I am to lazy to do. """
        ftr_count = X.shape[1]
        to_delete_n = self.features_fraction * ftr_count
        to_delete_n = int(to_delete_n)
        to_delete_indices = np.random.permutation(list(range(ftr_count)))[:to_delete_n]
        X[:, to_delete_indices] = 0
        return X


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.3, shuffle=True,
                                                        random_state=10)
    print(len(X_train))
    rf = RandomRegressionForest(100, 0.3, 0.3, 4)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    rf = RandomForestRegressor(100, max_depth=4, random_state=42, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
