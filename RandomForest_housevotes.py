
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statistics import mean, mode

class Node():
    def __init__(self):
        self.isNumFeat = False
        self.isLeafNode = False
        self.edges = {}


class DecisionTreeClassifier():
    def __init__(self, min_samples_split, max_depth, n_features):
        self.min_instances = min_samples_split

    def fit(self, df, feature_meta):
        self.features = list(df.columns.drop('target'))
        self.meta = feature_meta
        self.data = df
        self.tree = self._build_tree(df, 0)

    def fetch_major_class(self, df):
        return df["target"].value_counts().idxmax()

    def _build_tree(self, df, depth):

        #         if len(df) ==0:#<= self.min_instances:
        #             node = Node()
        #             node.isLeafNode = True
        #             node.leafValue = self.fetch_major_class(df)
        #             return node

        # print('------')

        # print(df["target"].value_counts())
        counts = dict(df["target"].value_counts())
        # print("match", max(counts.values()), len(df), counts)
        if max(counts.values()) == len(df):
            node = Node()
            node.isLeafNode = True
            node.leafValue = self.fetch_major_class(df)
            return node

        if len(self.features) == 0:
            node = Node()
            node.isLeafNode = True
            node.leafValue = self.fetch_major_class(df)
            return node

        best_attr, threshold = self._find_best_split(df, self.meta)
        ##print(best_attr)
        self.features.remove(best_attr)
        if self.meta[best_attr] == "NUMERICAL":
            parent_node = Node()
            parent_node.feature = best_attr
            parent_node.isNumFeat = True
            parent_node.threshold = threshold
            df_lt = df[df[best_attr] <= threshold]
            df_gt = df[df[best_attr] > threshold]

            node_lt = Node()
            if len(df_lt) == 0:
                node_lt.isLeafNode = True
                node.leafValue = self.fetch_major_class(df)
            else:
                node_lt = self._build_tree(df_lt, depth + 1)
            parent_node.edges['lt'] = node_lt

            node_gt = Node()
            if len(df_gt) == 0:
                node_gt.isLeafNode = True
                node_gt.leafValue = self.fetch_major_class(df)
            else:
                node_gt = self._build_tree(df_gt, depth + 1)
            parent_node.edges['gt'] = node_gt
            return parent_node


        else:
            feat_values = self.data[best_attr].unique()
            parent_node = Node()
            parent_node.feature = best_attr
            for f_value in feat_values:
                new_df = df[df[best_attr] == f_value]
                node = Node()
                if len(new_df) == 0:
                    node.isLeafNode = True
                    node.leafValue = self.fetch_major_class(df)
                else:
                    node = self._build_tree(new_df, depth + 1)
                parent_node.edges[f_value] = node
            return parent_node

    def predict(self, test_df):
        temp = self.tree
        while not temp.isLeafNode:
            feature_val = test_df[temp.feature]
            if self.meta[temp.feature] == 'NUMERICAL':
                if feature_val <= temp.threshold:
                    temp = temp.edges['lt']
                else:
                    temp = temp.edges['gt']
            else:
                feature_val = test_df[temp.feature]
                temp = temp.edges[feature_val]
        return temp.leafValue

    def get_entropy(self, dataset):
        probability_vector = dataset['target'].value_counts().div(len(dataset))
        entropy = probability_vector.apply(lambda x: -(x * np.log2(x))).sum()
        return entropy

    def _find_best_split(self, df, attributes_dict):
        ig = {}
        for feature in self.features:
            entropy_parent = self._get_entropy(df)
            if attributes_dict[feature] == 'CATEGORICAL':
                entropy_feature_split = self._get_entropy_feature(df, feature)
                info_gain = entropy_parent - entropy_feature_split
                ig[info_gain] = [feature, None]
            else:
                entropy_feature_split, feature_threshold = self._get_numerical_entropy_feature(df, feature)
                info_gain = entropy_parent - entropy_feature_split
                ig[info_gain] = [feature, feature_threshold]
        # returns list - [feature_name , threshold/None]
        return ig[max(list(ig.keys()))]

    def _get_entropy(self, df):
        # Entropy of parent
        entropy = 0
        for target in np.unique(df['target']):
            fraction = df['target'].value_counts()[target] / len(df['target'])
            entropy += -fraction * np.log2(fraction)
        return entropy

    def _get_numerical_entropy_feature(self, df, feature):
        entropy = 0
        unique_label_values = df['target'].unique()  # 0,1
        threshold = df[feature].mean()
        entropy_of_label = 0
        for label_value in unique_label_values:
            tdf = df[df[feature] <= threshold]
            num = len(tdf[tdf['target'] == label_value])
            den = len(tdf)
            if den == 0:
                entropy_of_label -= 0
            else:
                prob = num / den
                entropy_of_label -= prob * np.log2(prob)

        weight = den / len(df)
        entropy += weight * entropy_of_label

        entropy_of_label = 0
        for label_value in unique_label_values:
            tdf = df[df[feature] > threshold]
            num = len(tdf[tdf['target'] == label_value])
            den = len(tdf)
            if den != 0:
                prob = num / den
                entropy_of_label -= prob * np.log2(prob)
            else:
                entropy_of_label -= 0

        weight = den / len(df)
        entropy += weight * entropy_of_label

        return entropy, threshold

    def _get_entropy_feature(self, df, feature):
        entropy = 0
        unique_feature_values = df[feature].unique()  # 0,1,2
        unique_label_values = df['target'].unique()  # 0,1

        for feature_value in unique_feature_values:
            entropy_of_label = 0
            for label_value in unique_label_values:
                tdf = df[df[feature] == feature_value]
                num = len(tdf[tdf['target'] == label_value])
                den = len(tdf)

                prob = num / den
                entropy_of_label += prob * np.log2(prob)

            weight = den / len(df)
            entropy += weight * entropy_of_label
        return abs(entropy)


class RandomForest:
    def __init__(self, num_trees=25, min_samples_split=6, max_depth=5, n_feat=4):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []
        self.n_feat = n_feat

    @staticmethod
    def _sample(df):
        n = df.shape[0]
        sample_indices = np.random.randint(n, size=n)
        return df.iloc[sample_indices]

    def fit(self, df, meta):
        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        # Build each tree of the forest
        num_built = 0
        while num_built < self.num_trees:
            clf = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_feat
            )
            # Obtain data sample
            _df = self._sample(df)

            # Train
            clf.fit(_df, meta)
            # Save the classifier
            self.decision_trees.append(clf)
            num_built += 1
        ##print(self.decision_trees)

    def predict(self, df):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(df))
        return mode(y)


def kfold(data, k, target_col):
    # print("asd", data.columns)
    # Separate the data based on the target class label
    classes = np.unique(data[target_col])
    class_data = {}
    for c in classes:
        class_data[c] = data[data[target_col] == c]

    distribution_per_fold = {c: len(class_data[c]) / k for c in classes}
    k_fold = []
    for k_val in range(k):

        k_data = pd.DataFrame()
        for c in classes:
            if k_val == k - 1:
                sliced_data = class_data[c]
            else:
                sliced_data = class_data[c].sample(
                    n=int(distribution_per_fold[c]))
            k_data = pd.concat([k_data, sliced_data])
            class_data[c] = class_data[c].drop(sliced_data.index)
        k_fold.append(k_data)
    return k_fold


def main():
    df = pd.read_csv('house_votes_84.csv')
    df = df.rename(columns={"class": "target"})
    feat_meta = {}
    for c in df.columns:
        feat_meta[c] = "CATEGORICAL"

    num_folds = 10
    n_tree_choices = [1, 5, 10, 20, 30, 40, 50]

    # Create a KFold object
    kf = kfold(df, num_folds, 'target')

    # Loop through each fold and split the dataframe
    n_to_accuracies = {}
    n_to_precision = {}
    n_to_recall = {}
    n_to_f1 = {}

    # Stratified K cross validation

    for n in n_tree_choices:
        print("n tree ", n)
        n_to_accuracies[n] = []
        n_to_precision[n] = []
        n_to_recall[n] = []
        n_to_f1[n] = []

        for k in range(len(kf)):
            test_set = kf[k]

            train_set = pd.DataFrame()
            for i in range(len(kf)):
                if i != k:
                    train_set = pd.concat([train_set, kf[i]])

            cols = len(df.columns) - 1
            model = RandomForest(num_trees=n, n_feat=int(np.sqrt(cols)))
            model.fit(train_set, feat_meta)

            y_true = test_set['target']
            y_pred = []
            for idx, t in test_set.iterrows():
                y_test_pred = model.predict(t)
                y_pred.append(y_test_pred)

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            n_to_accuracies[n].append(accuracy)
            n_to_precision[n].append(precision)
            n_to_recall[n].append(recall)
            n_to_f1[n].append(f1)

        print("Accuracy:", np.mean(n_to_accuracies[n]))
        print("Precision:", np.mean(n_to_precision[n]))
        print("Recall:", np.mean(n_to_recall[n]))
        print("F1 score:", np.mean(n_to_f1[n]))

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for n in n_tree_choices:
        accuracy_list.append(np.mean(n_to_accuracies[n]))
        precision_list.append(np.mean(n_to_precision[n]))
        recall_list.append(np.mean(n_to_recall[n]))
        f1_score_list.append(np.mean(n_to_f1[n]))

    plt.plot(n_tree_choices, accuracy_list)
    plt.scatter(n_tree_choices, accuracy_list)
    plt.xlabel("No of trees")
    plt.ylabel("Accuracy")
    plt.title("No. of Trees V/S Accuracy")
    plt.show()

    plt.plot(n_tree_choices, precision_list)
    plt.scatter(n_tree_choices, precision_list)
    plt.xlabel("No of trees")
    plt.ylabel("Precision")
    plt.title("No. of Trees V/S Precision")
    plt.show()

    plt.plot(n_tree_choices, recall_list)
    plt.scatter(n_tree_choices, recall_list)
    plt.xlabel("No of trees")
    plt.ylabel("Recall")
    plt.title("No. of Trees V/S Recall")
    plt.show()

    plt.plot(n_tree_choices, f1_score_list)
    plt.scatter(n_tree_choices, f1_score_list)
    plt.xlabel("No of trees")
    plt.ylabel("F1")
    plt.title("No. of Trees V/S F1")
    plt.show()


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    main()
