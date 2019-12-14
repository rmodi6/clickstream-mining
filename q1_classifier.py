import argparse
import copy
import os.path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score

sys.setrecursionlimit(100000)


def train_test_split(train_file, test_file, y_train_file, y_test_file):
    """
    Method to create dataframes for training examples, training labels, test examples and test labels
    @param train_file: training file name
    @param test_file: test file name
    @param y_train_file: training labels file name
    @param y_test_file: test labels file name
    @return: dataframes for training examples, training labels, test examples and test labels
    """
    X_train = pd.read_csv(train_file, header=None, sep='\s+')
    X_test = pd.read_csv(test_file, header=None, sep='\s+')
    y_train = pd.read_csv(y_train_file, header=None, sep='\s+')[0]
    y_test = None
    if os.path.isfile(y_test_file):
        y_test = pd.read_csv(y_test_file, header=None, sep='\s+')[0]
    return X_train, X_test, y_train, y_test


class TreeNode:
    """
    Helper class used as a Node object in the Decision Tree
    self.val stores the attribute (column number) to split on in the decision tree
    self.children is a dictionary which points to the child node for a given attribute value
    """

    def __init__(self, data='T', children=None):
        """
        init method to initialize members of the class
        @param data: attribute value to split on, 'T' for leaf node with value 1 and 'F' for leaf node with value 0
        @param children: list containing node objects of children nodes
        """
        if children is None:
            children = [-1] * 5
        self.nodes = list(children)
        self.data = data


class ID3DecisionTree:
    """
    ID3DecisionTree model class to run the ID3 decision tree algorithm with chi-squared stopping criterion
    self.root points to the head node of the decision tree created by the fit method.
    """

    def __init__(self, p_value_threshold):
        """
        init method to initialize members of the class
        @param p_value_threshold: the p_value threshold to be used with the chi squared p_value
        """
        self.leaf_nodes_count = 0  # Count of leaf nodes
        self.internal_nodes_count = 0  # Count of internal nodes
        self.p_value_threshold = p_value_threshold  # Significance level threshold for pvalue
        self.root = None  # Root node of the decision tree

    def fit(self, X, y, attributes=None, parent=None, attr_val=None):
        """
        This method constructs a decision tree using the ID3 algorithm for the training examples X having target values
        y. It uses the p_value_threshold for the chi-squared stopping criterion.
        @param X: training examples
        @param y: labels of the training examples
        @param attributes: columns in the training examples to be considered at the current node in the tree
        @param parent: parent node which calls this method recursively
        @param attr_val: value of the attribute of the parent node which calls this method recursively
        @return: object instance of the ID3Decision class
        """
        if attributes is None:  # If attributes is not set
            attributes = list(X.columns)  # Set attributes to be columns of the training dataset
        if 0 not in y.unique():  # If all examples are label 1
            curr_node = TreeNode(data='T')  # Create a node with label 1
            self.leaf_nodes_count += 1  # Increment leaf node count
        elif 1 not in y.unique():  # If all examples are label 0
            curr_node = TreeNode(data='F')  # Create a node with label 0
            self.leaf_nodes_count += 1  # Increment leaf node count
        elif len(attributes) == 0:  # If there are no attributes left to split on
            curr_node = self.create_max_label_node(y)  # Create a node with label = most common value of the target y
        else:
            # Get the attribute with the least entropy i.e. maximum information gain. This is the attribute which we
            # will split at the current node in the tree
            attr_to_split = self.least_entropy_attr(X[attributes], y)
            attributes.remove(attr_to_split)  # Remove the selected attribute from the attributes list

            # Compute the p_value using the chi-squared distribution
            p_value = self.compute_chisquare(X[attr_to_split], y)

            if p_value < self.p_value_threshold:  # If the p_value is less than the significance level
                curr_node = TreeNode(data=attr_to_split + 1)  # Create a node with value as best attribute to split on
                self.internal_nodes_count += 1  # Increment internal node count

                for category in range(1, 6):  # For every unique value in the attribute we are splitting
                    # Create a new training example new_X where the attribute only consists of the current value
                    new_X = X[X[attr_to_split] == category]
                    if new_X.empty:  # If there are no rows in this new training example
                        # Create a node with label = most common value of the target y
                        curr_node.nodes[category - 1] = self.create_max_label_node(y)
                    else:
                        # Create a new label dataset which has rows corresponding to the new training example new_X
                        new_y = copy.deepcopy(y.loc[new_X.index])
                        # Make a recursive call to the fit method which will create a subtree under the
                        # attribute_to_split node and add the subtree's head node to the children of the current node
                        self.fit(new_X, new_y, attributes, parent=curr_node, attr_val=category - 1)
            else:  # If the p_value is greater than equal to the significance level
                # Create a node with label = most common value of the target y
                curr_node = self.create_max_label_node(y)
        # Handle the recursive call to the fit method
        if parent is None:  # If parent is None then this is the first call to the fit method
            self.root = curr_node  # Assign current node as the root node of the decision tree
        else:  # If parent is not None then this is a recursive call to the fit method called by parent node
            parent.nodes[attr_val] = curr_node  # Added the current node to the children of the parent node
        return self

    def create_max_label_node(self, y):
        """
        Creates and returns a tree node with label = most common value of the target y
        @param y: target values
        @return: TreeNode with label = most common value of the target y
        """
        self.leaf_nodes_count += 1  # Increment leaf node count
        y_counts = y.value_counts().to_dict()  # Counts occurrences of 1's and 0's
        if y_counts.get(1, 0) >= y_counts.get(0, 0):  # If count of 1's is greater than or equal to 0's
            curr_node = TreeNode(data='T')  # Create node with label 1
        else:
            curr_node = TreeNode(data='F')  # Create node with label 0
        return curr_node

    def least_entropy_attr(self, X, y):
        """
        Compute the least entropy attribute. This is equivalent to finding the attribute with max information gain
        @param X: training examples with columns as attributes
        @param y: target values of the training examples
        @return: attribute (column number) having least entropy
        """
        entropies = {}
        for attr in X.columns:  # For each attribute in the training data
            entropies[attr] = self.compute_entropy(X[attr], y)  # Compute and store the entropy of the attribute
        return min(entropies, key=entropies.get)  # Return attribute with min entropy out of all attributes

    @staticmethod
    def compute_entropy(attr_values, target_values):
        """
        Compute the entropy of an attribute (column)
        @param attr_values: all the values of the attribute
        @param target_values: target values for the given attribute values
        @return: entropy of given attribute values
        """
        entropy = 0
        total_count = len(attr_values)
        for category in attr_values.unique():  # For each unique value in the attribute
            category_values = attr_values[attr_values == category]  # Get the values of the current loop
            category_count = float(len(category_values))  # Get the count of all the values
            category_target_values = target_values.loc[category_values.index]  # Get the corresponding target values
            y_counts = category_target_values.value_counts().to_dict()  # Get counts of corresponding target values
            if len(y_counts) != 2:  # If the target values contains only one value (either 0 or 1)
                e = 0  # Set entropy for current value of attribute as 0
            else:  # If target values contain both 0's and 1's
                p = y_counts[1] / category_count  # Calculate ratio of positive count p
                n = y_counts[0] / category_count  # Calculate ratio of negative count n
                e = -(p * np.log2(p) + n * np.log2(n))  # Calculate entropy for current value of attribute using p and n
            entropy += (category_count / total_count) * e  # Update the entropy of attribute for current value
        return entropy  # Return the entropy for given attribute

    @staticmethod
    def compute_chisquare(attr_values, target_values):
        """
        Compute the p_value using the chisquared distribution
        @param attr_values: all the values of the attribute
        @param target_values: target values for the given attribute values
        @return: p_value
        """
        f_exp, f_obs = [], []
        target_value_counts = target_values.value_counts().to_dict()  # Calculate counts of target values 0's and 1's
        p = target_value_counts.get(1, 0)  # Count of positive 1's examples
        n = target_value_counts.get(0, 0)  # Count of negative 0's examples
        N = p + n  # Total number of examples
        for category in attr_values.unique():  # For each unique value in the attribute
            category_values = attr_values[attr_values == category]  # Get the values of the current loop
            T_i = float(len(category_values))  # Get the count of all the values
            category_target_values = target_values.loc[category_values.index]  # Get the corresponding target values
            y_counts = category_target_values.value_counts().to_dict()  # Get counts of corresponding target values
            p_i, n_i = y_counts.get(1, 0), y_counts.get(0, 0)  # Count number of 1's and 0's in target values
            # Calculate the expected and observed number of positives and negatives
            p_prime_i = p * T_i / N
            n_prime_i = n * T_i / N
            if p_prime_i != 0:
                f_exp.append(p_prime_i)
                f_obs.append(p_i)
            if n_prime_i != 0:
                f_exp.append(n_prime_i)
                f_obs.append(n_i)
        score, p_value = chisquare(f_obs, f_exp)  # Calculate p_value using scipy.stats.chisquare
        return p_value  # Return the p_value for the given attribute

    def save(self, model_path):
        """
        Save the ID3DecisionTree model at the given path using pickle dump
        @param model_path: path where the model is to be saved
        @return: None
        """
        with open(model_path, 'w') as f:
            pickle.dump(self.root, f)

    def predict(self, X):
        """
        Method to compute predictions for test data X
        @param X: test data whose target values are to be predicted
        @return: predictions for test data
        """
        # Call predict_label method for each row in the test dataset and store its output in a pandas data series
        y_pred = pd.Series([self.predict_label(self.root, row) for row in X[X.columns].values])
        return y_pred  # Return the test predictions

    def predict_label(self, node, row):
        """
        Recursive method to compute prediction for a test data row by traversing the Decision Tree
        @param node: current node of the tree in the traversal
        @param row: test data row whose target value is to be predicted
        @return: predicted target value for the given row
        """
        if node.data == 'T':  # If node is a leaf node
            return 1  # Return the label of the leaf node
        if node.data == 'F':
            return 0
        # Recursively call predict_label method on the child node obtained from the attribute value of the current node
        return self.predict_label(node.nodes[row[int(node.data) - 1] - 1], row)


if __name__ == '__main__':
    # Argument parser to parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', dest='p_value_threshold', action='store', type=float, help='pvalue threshold')
    parser.add_argument('-f1', dest='train_dataset', action='store', type=str, help='train_dataset')
    parser.add_argument('-f2', dest='test_dataset', action='store', type=str, help='test_dataset')
    parser.add_argument('-o', dest='output_file', action='store', type=str, help='output_file')
    parser.add_argument('-t', dest='decision_tree', action='store', type=str, help='decision_tree')

    args = parser.parse_args()

    # Get the filenames for the training and testing data along with the filenames of their target values data
    train_file, test_file = str(args.train_dataset), str(args.test_dataset)
    y_train_file, y_test_file = train_file.replace('.csv', '_label.csv'), test_file.replace('.csv', '_label.csv')
    # Get the dataframes for training and testing data
    X_train, X_test, y_train, y_test = train_test_split(train_file, test_file, y_train_file, y_test_file)

    # Create a ID3DecisionTree model and fit the model for the given training data
    model = ID3DecisionTree(args.p_value_threshold).fit(X_train, y_train)
    model.save(args.decision_tree)  # Save the model at the specified location

    preds = model.predict(X_test)  # Get the predictions for the test dataset
    preds.to_csv(args.output_file, header=False, index=False)  # Write the predictions to the output file

    print('Number of internal nodes: {} \nNumber of leaf nodes: {}'.format(model.internal_nodes_count,
                                                                           model.leaf_nodes_count))
    if y_test is not None:
        accuracy = accuracy_score(y_test, preds)  # Calculate accuracy if the target values for test data is given
        print('Model Accuracy: {}'.format(accuracy))
