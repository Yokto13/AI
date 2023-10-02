from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class Node:
    ftr_index: int | None
    ftr_val: float | None
    is_leaf: bool
    target: float | None
    left: Optional["Node"]
    right: Optional["Node"]

    def predict(self, x: np.array):
        if self.is_leaf: return self.target
        if x[self.ftr_index] <= self.ftr_val:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class RegressionTree(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root: Node | None = None

    def build_tree(self, X: np.array, y: np.array, depth) -> Optional[Node]:
        if depth == self.max_depth or len(X) == 1:
            target = y.mean()
            return Node(
                ftr_index=None,
                ftr_val=None,
                is_leaf=True,
                target=target,
                left=None,
                right=None
            )
        ftr_index, ftr_val, X, y = self.find_best_split(X, y)
        if ftr_index is None:
            target = y.mean()
            return Node(
                ftr_index=None,
                ftr_val=None,
                is_leaf=True,
                target=target,
                left=None,
                right=None
            )
        X_left, y_left, X_right, y_right = self.split_data(X, y, ftr_index, ftr_val)
        return Node(
            ftr_index=ftr_index,
            ftr_val=ftr_val,
            is_leaf=False,
            target=None,
            left=self.build_tree(X_left, y_left, depth + 1),
            right=self.build_tree(X_right, y_right, depth + 1)
        )

    def find_best_split(self, X: np.array, y: np.array) -> tuple[int, float, np.array, np.array]:
        var = lambda p2, p1, cnt: (p2 / cnt - (p1 / cnt) ** 2) * (cnt / len(X))
        Split = namedtuple("Split", ['var', 'ftr_val', 'ftr_idx'])
        best_result = Split(np.inf, None, None)
        y_squared = np.power(y, 2)
        no_split_var = var(np.sum(y_squared), np.sum(y), len(y))
        for ftr_idx in range(X.shape[-1]):
            sorted_inds = np.argsort(X[:, ftr_idx])
            X, y, y_squared = X[sorted_inds], y[sorted_inds], y_squared[sorted_inds]
            left_squared, right_squared = 0, np.sum(y_squared)
            left_sum, right_sum = 0, np.sum(y)
            ftr_vals = sorted([f for f in set(ff for ff in X[:, ftr_idx])])
            for left_size, ftr_val in enumerate(X[:len(X) - 1, ftr_idx]):
                left_squared += y_squared[left_size]
                right_squared -= y_squared[left_size]
                left_sum += y[left_size]
                right_sum -= y[left_size]
                left_var = var(left_squared, left_sum, left_size + 1)
                right_var = var(right_squared, right_sum, len(y) - left_size - 1)
                tot_var = left_var + right_var
                if tot_var < best_result.var and tot_var < no_split_var and ftr_vals.index(ftr_val) < len(ftr_vals) - 1:
                    best_result = Split(tot_var, (ftr_val + ftr_vals[ftr_vals.index(ftr_val) + 1]) / 2, ftr_idx)
        if best_result.ftr_idx is not None:
            sorted_inds = np.argsort(X[:, best_result.ftr_idx])
            X, y = X[sorted_inds], y[sorted_inds]
        return best_result.ftr_idx, best_result.ftr_val, X, y

    def split_data(self, X, y, ftr_index, ftr_val):
        X_left, y_left, X_right, y_right = [], [], [], []
        for x, t in zip(X, y):
            if x[ftr_index] <= ftr_val:
                X_left.append(x)
                y_left.append(t)
            else:
                X_right.append(x)
                y_right.append(t)
        return list(map(np.array, (X_left, y_left, X_right, y_right)))

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)

    def predict(self, X):
        return [self.root.predict(x) for x in X]


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.3, shuffle=True,
                                                        random_state=10)
    print(len(X_train))
    rt = RegressionTree(5)
    rt.fit(X_train, y_train)
    y_pred = rt.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    dtr = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
