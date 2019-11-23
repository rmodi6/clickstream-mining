from scipy.stats import chisquare
import argparse
import numpy as np
import pandas as pd


def train_test_split(train_file, test_file, y_train_file, y_test_file):
    X_train = pd.read_csv(train_file, header=None, sep='\s+')
    X_test = pd.read_csv(test_file, header=None, sep='\s+')
    y_train = pd.read_csv(y_train_file, header=None, sep='\s+')[0]
    y_test = pd.read_csv(y_test_file, header=None, sep='\s+')[0]
    return X_train, X_test, y_train, y_test


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []


class ID3DecisionTree:
    def __init__(self, p_value_threshold):
        self.p_value_threshold = p_value_threshold
        self.root = None

    def fit(self, X, y, parent=None):
        if 0 not in y.unique():
            curr_node = TreeNode(val=True)
        elif 1 not in y.unique():
            curr_node = TreeNode(val=False)
        elif len(X.columns) == 0:
            curr_node = self.create_max_label_node(y)
        else:
            attr_to_split = self.least_entropy_attr(X, y)
            p_value = self.chisquare_split(X, y)

            if p_value < self.p_value_threshold:
                curr_node = TreeNode(val=attr_to_split + 1)

                for category in X[attr_to_split].unique():
                    new_X = X[X[attr_to_split] == category]
                    if new_X.empty:
                        curr_node = self.create_max_label_node(y)
                    else:
                        new_X.drop(attr_to_split, axis=1, inplace=True)
                        new_y = y.iloc[new_X.index]
                        self.fit(new_X, new_y, parent=curr_node)
            else:
                curr_node = self.create_max_label_node(y)
        if parent is None:
            self.root = curr_node
        else:
            parent.children.append(curr_node)
        return self

    @staticmethod
    def create_max_label_node(y):
        y_counts = y.value_counts().to_dict()
        if y_counts[1] > y_counts[0]:
            curr_node = TreeNode(val=True)
        else:
            curr_node = TreeNode(val=False)
        return curr_node

    def least_entropy_attr(self, X, y):
        entropies = {}
        for attr in X.columns:
            entropies[attr] = self.compute_entropy(X[attr], y)
        return min(entropies, key=entropies.get)

    @staticmethod
    def compute_entropy(attr_values, target_values):
        entropy = 0
        total_count = len(attr_values)
        for category in attr_values.unique():
            category_values = attr_values[attr_values == category]
            category_count = float(len(category_values))
            category_target_values = target_values.iloc[category_values.index]
            y_counts = category_target_values.value_counts().to_dict()
            if len(y_counts) != 2:
                e = 0
            else:
                p = y_counts[1] / category_count
                n = y_counts[0] / category_count
                e = -(p * np.log2(p) + n * np.log2(n))
            entropy += (category_count / total_count) * e
        return entropy

    @staticmethod
    def chisquare_split(X, y):
        return 1

    def predict(self, X):
        pass
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', dest='p_value_threshold', action='store', type=float, help='pvalue')
    parser.add_argument('-f1', dest='train_dataset', action='store', type=str, help='train_dataset')
    parser.add_argument('-f2', dest='test_dataset', action='store', type=str, help='test_dataset')
    parser.add_argument('-o', dest='output_file', action='store', type=str, help='output_file')
    parser.add_argument('-t', dest='decision_tree', action='store', type=str, help='decision_tree')

    args = parser.parse_args()

    train_file, test_file = str(args.train_dataset), str(args.test_dataset)
    y_train_file, y_test_file = train_file.replace('feat', 'labs'), test_file.replace('feat', 'labs')
    X_train, X_test, y_train, y_test = train_test_split(train_file, test_file, y_train_file, y_test_file)

    model = ID3DecisionTree(args.p_value_threshold).fit(X_train, y_train)

    y_pred = model.predict(X_test)
