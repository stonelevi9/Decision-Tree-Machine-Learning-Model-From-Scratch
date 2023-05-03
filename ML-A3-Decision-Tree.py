import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
y = y.reshape((150, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)


# This is our node class, and it has a constructor that allows us to create node objects with information we need to
# store in them.
class Node:
    def __init__(self, value=None, x_data=None, y_data=None, left=None, right=None, info_gain=None, threshold=None,
                 index=None):
        self.value = value
        self.x_data = x_data
        self.y_data = y_data
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.threshold = threshold
        self.index = index


# This is our create_tree method, and it is responsible for recursively creating our tree while calling our other logic
# methods at the appropriate points per call.
def create_tree(node, x_data, y_data):
    rows, columns = np.shape(x_data)
    info_gain, threshold, index, x_left, x_right, y_left, y_right = find_split(x_data, y_data, columns, rows)
    if info_gain > 0:
        left_node = None
        right_node = None
        left_child = create_tree(left_node, x_left, y_left)
        right_child = create_tree(right_node, x_right, y_right)
        return Node(x_data=x_data, y_data=y_data, left=left_child, right=right_child, info_gain=info_gain,
                    threshold=threshold, index=index)
    v = list(y_data)
    y = max(v, key=v.count)
    return Node(value=y)


# This is our calc_entropy method and as the name suggests, it is responsible for calculating the entropy of a given
# data set.
def calc_entropy(y_subset):
    data_entries, data_features = np.shape(y_subset)
    target, count = np.unique(y_subset, return_counts=True)
    entropy = 0
    for i in range(len(target)):
        proportion = count[i] / data_entries
        entropy = entropy + (proportion * np.log2(proportion))
    entropy = -entropy
    return entropy


# This is our calc_info_gain method, and it is responsible for calculating info gain given the parent dataset, and both
# the left and the right child dataset.
def calc_info_gain(y_dataset, y_left, y_right):
    w1 = len(y_left) / len(y_dataset)
    w2 = len(y_right) / len(y_dataset)
    info_gain = calc_entropy(y_dataset) - (w1 * calc_entropy(y_left) + w2 * calc_entropy(y_right))
    return info_gain


# This is our find_split method, and it is responsible for finding the split that has the highest information gain.
# It does this by going through every unique value of a feature and splitting the dataset, and then it calculates the
# information gain on that iteration. It keeps track of the split with the best information and returns the relevant
# information associated with it.
def find_split(x_data, y_data, data_features, data_entries):
    max_info_gain = -9999999999
    current_threshold = 0
    current_index = 0
    current_x_left = []
    current_x_right = []
    current_y_left = []
    current_y_right = []
    for i in range(data_features):
        feature_column = x_data[:, i]
        all_splits = np.unique(feature_column)
        for test_split in all_splits:
            x_left, y_left, x_right, y_right = split(x_data, y_data, test_split, i, data_entries)
            if len(x_left) <= 0 or len(x_right) <= 0:
                continue
            current_gain = calc_info_gain(y_data, y_left, y_right)
            if current_gain > max_info_gain:
                max_info_gain = current_gain
                current_threshold = test_split
                current_index = i
                current_x_left = x_left
                current_x_right = x_right
                current_y_left = y_left
                current_y_right = y_right
    return max_info_gain, current_threshold, current_index, current_x_left, current_x_right, current_y_left, current_y_right


# This is our split method, and it is responsible for splitting a dataset into two datasets around a threshold.
def split(x_dataset, y_dataset, split_value, index, data_entries):
    x_left = np.empty((0, 4))
    x_right = np.empty((0, 4))
    y_left = np.empty((0, 1))
    y_right = np.empty((0, 1))
    for i in range(data_entries):
        new_x_row = x_dataset[i]
        new_y_row = y_dataset[i]
        if x_dataset[i][index] <= split_value:
            x_left = np.vstack([x_left, new_x_row])
            y_left = np.vstack([y_left, new_y_row])
        else:
            x_right = np.vstack([x_right, new_x_row])
            y_right = np.vstack([y_right, new_y_row])
    return x_left, y_left, x_right, y_right


# This is our predict method, and it is responsible for finding the corresponding position in the tree given a test set
# entry to predict upon. Once it reaches a leaf node, it returns this leaf node's value.
def predict(test, node):
    if node.value is not None:
        return node.value
    value = test[node.index]
    if value <= node.threshold:
        return predict(test, node.left)
    else:
        return predict(test, node.right)


# This is our testing method, and it essentially just calls our predict method on every entry in our test set and
# returns an array containing all the predictions.
def testing(x_set, node):
    test_rows, test_cols = np.shape(x_set)
    predictions = np.empty((0, 1))
    for i in range(test_rows):
        predictions = np.vstack([predictions, predict(x_set[i], node)])
    return predictions


# This is our main method. It creates our root node and then calls create_tree with that node. Lastly, it tests our
# test set and prints out our accuracy score.
def main():
    root = None
    root = create_tree(root, x_train, y_train)
    predicted = testing(x_test, root)
    accuracy = accuracy_score(y_test, predicted)
    print('accuracy score: ', accuracy)


if __name__ == '__main__':
    main()
