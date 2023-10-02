from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Generator

import numpy as np


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


class ClassificationTree:
    def __init__(self, n_of_classes: int, max_depth: int,
                 ftr_generator: Generator[int, int, None] = range):
        self.n_of_classes = n_of_classes
        self.max_depth = max_depth
        self.root: Node | None = None
        self.ftr_generator = ftr_generator

    def target_to_pd(self, y):
        pds = np.zeros((y.size, self.n_of_classes))
        pds[np.arange(y.size), y] = 1
        return np.average(pds, axis=0)

    def build_tree(self, X: np.array, y: np.array, depth) -> Optional[Node]:
        if depth == self.max_depth or len(X) == 1:
            target = self.target_to_pd(y)
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
            target = self.target_to_pd(y)
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
        def entropy(y):
            unique, counts = np.unique(y, return_counts=True)
            tot = sum(counts)
            return - sum((x / tot * np.log(x) for x in counts))

        Split = namedtuple("Split", ['score', 'ftr_val', 'ftr_idx'])
        no_split_score = entropy(y)
        best_result = Split(no_split_score, None, None)
        for ftr_idx in self.ftr_generator(X.shape[-1]):
            sorted_inds = np.argsort(X[:, ftr_idx])
            for i in range(len(sorted_inds) - 1):
                if X[i, ftr_idx] == X[i + 1, ftr_idx]:
                    continue
                y_left, y_right = y[sorted_inds[:i + 1]], y[sorted_inds[i + 1:]]
                score_left, score_right = entropy(y_left), entropy(y_right)
                split_score = score_left + score_right
                if split_score < best_result.score:
                    val = (X[i, ftr_idx] + X[i + 1, ftr_idx]) / 2
                    best_result = Split(split_score, val, ftr_idx)
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

    def hard_predict(self, X):
        return [np.argmax(x) for x in self.predict(X)]


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeClassifier

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.3, shuffle=True,
                                                        random_state=10)
    print(len(X_train))
    rt = ClassificationTree(y_train.max() + 1, 5)
    rt.fit(X_train, y_train)
    pres = rt.predict(X_test)
    y_pred = np.argmax(rt.predict(X_test), axis=1)
    print(mean_squared_error(y_test, y_pred))
    dtr = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
