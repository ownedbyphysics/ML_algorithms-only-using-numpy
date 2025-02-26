class DecisionTree:
    def __init__(self, max_depth=5):
        """
        Initialize the DecisionTree classifier.
        :param max_depth: Maximum depth of the tree to prevent overfitting.
        """
        self.max_depth = max_depth
        self.tree = None
    
    def gini_impurity(self, y):
        """
        Computes the Gini impurity of a given set of labels.
        Gini impurity measures how often a randomly chosen element would be incorrectly labeled.
        It is measured before the split and then different splits are done in order to follow the one
        that minimizes the gini impurity. It tries to split into as many examples of the same class it can. 
        The code here returns [0-1] with 0 meaning perfect split of classes and 1 means balanced split.
        :return: Gini impurity score.
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)
    
    def best_split(self, X, y):
        """
        Finds the best feature and threshold to split the dataset.
        Iterrates through the features and tries each unique value as threshold
        The goal is to find the split that minimizes the Gini impurity, creating 
        subsets that are as pure as possible.

        Steps:
        1. Initialize `best_gini` with a high value, and `best_feature` and `best_threshold` as None.
        2. Loop through each feature (column in X).
        3. For each feature, get all unique values and consider them as possible thresholds.
        4. For each threshold:
           - Split the dataset into two groups: left (values ≤ threshold) and right (values > threshold).
           - Skip the threshold if it results in an empty left or right group.
           - Compute the Gini impurity for both groups.
           - Calculate the weighted average Gini impurity for the split.
           - If this split has a lower impurity than previous splits, update `best_gini`, `best_feature`, and `best_threshold`.
        5. Return the best feature and threshold found.
        :return: The best feature index and threshold value for the split.
        """
        m, n = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(n):
            thresholds = np.unique(X[:, feature])  
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue  # Skip invalid splits
                
                left_gini = self.gini_impurity(y[left_mask])
                right_gini = self.gini_impurity(y[right_mask])
                gini = (left_mask.sum() * left_gini + right_mask.sum() * right_gini) / m
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        :param X: Feature matrix.
        :param y: Target labels.
        :param depth: Current depth of the tree.
        :return: Tree structure (either a dictionary for a node or a class label for a leaf node).
        """
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))  # Return the most common class as the leaf node
        
        feature, threshold = self.best_split(X, y)
        if feature is None:
            return np.argmax(np.bincount(y))
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        """
        Train the decision tree by building its structure from the dataset.
        :param X: Feature matrix.
        :param y: Target labels.
        """
        self.tree = self.build_tree(X, y)
    
    def predict_single(self, x, node):
        """
        Traverse the tree for a single sample to make a prediction.
        :param x: Single sample features.
        :param node: Current node of the tree.
        :return: Predicted class label.
        """
        if isinstance(node, dict):  # If node is not a leaf
            if x[node['feature']] <= node['threshold']:
                return self.predict_single(x, node['left'])
            else:
                return self.predict_single(x, node['right'])
        return node  # Leaf node (class label)
    
    def predict(self, X):
        """
        Predict class labels for a set of input samples.
        :param X: Feature matrix.
        :return: Array of predicted class labels.
        """
        return np.array([self.predict_single(x, self.tree) for x in X])

# Example usage:
X = np.array([[2.7], [1.3], [3.5], [1.1], [2.2], [3.0], [1.9]])
y = np.array([0, 1, 0, 1, 0, 0, 1])

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
preds = tree.predict(X)
print("Predictions:", preds)
