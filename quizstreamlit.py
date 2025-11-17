import streamlit as st

# ===========================================
# FULL QUESTION BANK (120+ MCQs from your PDF)
# ===========================================
QUESTIONS = [
    {
        "question": "What does KDD stand for in data mining?",
        "options": [
            "Key Data Distribution",
            "Knowledge Discovery in Databases",
            "Kernel Data Design",
            "Known Data Determination"
        ],
        "answer": "B"
    },
    {
        "question": "Data mining is primarily the ______ step of the KDD process.",
        "options": [
            "first",
            "last",
            "analysis",
            "deployment"
        ],
        "answer": "C"
    },
    {
        "question": "Machine learning is considered a subset of:",
        "options": [
            "Database systems",
            "Statistics",
            "Artificial intelligence",
            "Data warehousing"
        ],
        "answer": "C"
    },
    {
        "question": "Which type of learning is NOT part of this Data Mining module?",
        "options": [
            "Supervised learning",
            "Unsupervised learning",
            "Reinforcement learning",
            "Semi-supervised learning"
        ],
        "answer": "C"
    },
    {
        "question": "According to the lecture, approximately what percentage of effort is typically spent on data cleaning and pre-processing?",
        "options": ["20%", "40%", "60%", "80%"],
        "answer": "C"
    },
    {
        "question": "In which CRISP-DM phase is the data split into training and test sets?",
        "options": [
            "Business Understanding",
            "Data Understanding",
            "Data Preparation",
            "Modelling"
        ],
        "answer": "C"
    },
    {
        "question": "Which ethical issue is highlighted using the example of zip code, birth date, and sex?",
        "options": [
            "Data storage costs",
            "Difficulty in anonymizing personal data",
            "Slow algorithm performance",
            "Lack of open-source tools"
        ],
        "answer": "B"
    },
    {
        "question": "The Apriori algorithm is used for which data mining task?",
        "options": [
            "Classification",
            "Regression",
            "Clustering",
            "Association rule mining"
        ],
        "answer": "D"
    },
    {
        "question": "Which of the following is a supervised learning task?",
        "options": [
            "Clustering",
            "Dimensionality reduction",
            "Classification",
            "Anomaly detection"
        ],
        "answer": "C"
    },
    {
        "question": "Numeric prediction is also known as:",
        "options": [
            "Classification",
            "Clustering",
            "Regression",
            "Association"
        ],
        "answer": "C"
    },
    {
        "question": "Clustering is a form of:",
        "options": [
            "Supervised learning",
            "Unsupervised learning",
            "Reinforcement learning",
            "Semi-supervised learning"
        ],
        "answer": "B"
    },
    {
        "question": "According to the lecture, 85% of Americans can be identified using:",
        "options": [
            "Name, phone number, and email",
            "Social security number and address",
            "Zip code, birth date, and sex",
            "IP address and device ID"
        ],
        "answer": "C"
    },
    {
        "question": "Which methodology is described as a 'step-by-step guide' used by IBM?",
        "options": [
            "KDD",
            "CRISP-DM",
            "ASUM",
            "Agile"
        ],
        "answer": "C"
    },
    {
        "question": "In supervised learning, the model is trained using:",
        "options": [
            "Only input features",
            "Input features and known target outputs",
            "No labels",
            "Only class distributions"
        ],
        "answer": "B"
    },
    {
        "question": "Which algorithm is used in unsupervised learning?",
        "options": [
            "Decision Tree",
            "Naïve Bayes",
            "K-means",
            "Logistic Regression"
        ],
        "answer": "C"
    },
    {
        "question": "What is the role of metadata in data mining?",
        "options": [
            "To increase storage efficiency",
            "To provide background knowledge that can restrict the search space",
            "To replace data cleaning",
            "To visualize high-dimensional data"
        ],
        "answer": "B"
    },
    {
        "question": "EDA stands for:",
        "options": [
            "Enhanced Data Algorithm",
            "Exploratory Data Analysis",
            "Efficient Data Aggregation",
            "Experimental Data Acquisition"
        ],
        "answer": "B"
    },
    {
        "question": "Which data type allows only equality comparisons and has no inherent order?",
        "options": ["Nominal", "Ordinal", "Interval", "Ratio"],
        "answer": "A"
    },
    {
        "question": "Temperature in degrees Celsius is an example of which data type?",
        "options": ["Nominal", "Ordinal", "Interval", "Ratio"],
        "answer": "C"
    },
    {
        "question": "Distance (e.g., in meters) is an example of which data type?",
        "options": ["Nominal", "Ordinal", "Interval", "Ratio"],
        "answer": "D"
    },
    {
        "question": "Which encoding method is appropriate for nominal categorical data to avoid implying order?",
        "options": [
            "Ordinal encoding",
            "Label encoding",
            "One-hot encoding",
            "Standardization"
        ],
        "answer": "C"
    },
    {
        "question": "What is the purpose of standardization?",
        "options": [
            "Scale features to [0, 1]",
            "Transform features to have mean = 0 and standard deviation = 1",
            "Remove outliers",
            "Encode categorical variables"
        ],
        "answer": "B"
    },
    {
        "question": "MinMaxScaler performs:",
        "options": [
            "Standardization",
            "Normalization",
            "PCA",
            "Binarization"
        ],
        "answer": "B"
    },
    {
        "question": "Outliers can be detected using:",
        "options": [
            "Boxplots",
            "Scatter plots",
            "Statistical methods",
            "All of the above"
        ],
        "answer": "D"
    },
    {
        "question": "According to the lecture, when should outliers be removed?",
        "options": [
            "Always",
            "Never",
            "Only if they result from data errors",
            "Only in supervised learning"
        ],
        "answer": "C"
    },
    {
        "question": "Univariate outlier detection examines:",
        "options": [
            "Combinations of features",
            "One feature at a time",
            "Only class labels",
            "Correlation matrices"
        ],
        "answer": "B"
    },
    {
        "question": "Multivariate outlier detection can be performed using:",
        "options": ["DBSCAN", "Histograms", "Mean imputation", "One-hot encoding"],
        "answer": "A"
    },
    {
        "question": "The IQR (Interquartile Range) is used in:",
        "options": [
            "Normalization",
            "Standardization",
            "Boxplot outlier detection",
            "PCA"
        ],
        "answer": "C"
    },
    {
        "question": "Missing values can result from:",
        "options": [
            "Equipment malfunction",
            "Changes in experimental design",
            "Collation of datasets",
            "All of the above"
        ],
        "answer": "D"
    },
    {
        "question": "Replacing missing values with the mode is appropriate for:",
        "options": [
            "Continuous data",
            "Ratio data",
            "Categorical data",
            "Interval data"
        ],
        "answer": "C"
    },
    {
        "question": "Why is it critical to record all pre-processing actions?",
        "options": [
            "To reduce file size",
            "For reproducibility and transparency",
            "To speed up training",
            "To avoid using Python"
        ],
        "answer": "B"
    },
    {
        "question": "One-hot encoding may lead to:",
        "options": [
            "Faster training",
            "Reduced dimensionality",
            "High dimensionality",
            "Improved interpretability"
        ],
        "answer": "C"
    },
    {
        "question": "Which algorithm is sensitive to unscaled features?",
        "options": [
            "Decision Tree",
            "Naïve Bayes",
            "K-Nearest Neighbors (KNN)",
            "Apriori"
        ],
        "answer": "C"
    },
    {
        "question": "For ordinal data like letter grades (A, B, C), the correct encoding approach is:",
        "options": [
            "One-hot encoding",
            "Manual mapping to integers preserving order",
            "Standardization",
            "Random assignment"
        ],
        "answer": "B"
    },
    {
        "question": "PCA is used primarily for:",
        "options": [
            "Outlier detection",
            "Feature scaling",
            "Dimensionality reduction",
            "Handling missing values"
        ],
        "answer": "C"
    },
    {
        "question": "Which pre-processing step is most important before applying K-means or SVM?",
        "options": [
            "One-hot encoding",
            "Standardization or normalization",
            "Deleting all outliers",
            "Converting to ordinal"
        ],
        "answer": "B"
    },
    {
        "question": "In a decision tree, internal nodes represent:",
        "options": [
            "Class labels",
            "Predicted values",
            "Attribute tests",
            "Training instances"
        ],
        "answer": "C"
    },
    {
        "question": "Leaf nodes in a classification tree output:",
        "options": [
            "Attribute splits",
            "Class probabilities or labels",
            "Gini scores",
            "Entropy values"
        ],
        "answer": "B"
    },
    {
        "question": "Which impurity measure is used by the CART algorithm?",
        "options": ["Entropy", "Information Gain", "Gini Index", "Chi-square"],
        "answer": "C"
    },
    {
        "question": "Information Gain is calculated as:",
        "options": [
            "Entropy before split + Entropy after split",
            "Entropy before split – Entropy after split",
            "Gini before split – Gini after split",
            "Accuracy difference"
        ],
        "answer": "B"
    },
    {
        "question": "Overfitting in decision trees can be reduced by:",
        "options": [
            "Increasing max_depth",
            "Decreasing min_samples_split",
            "Pruning",
            "Using more features"
        ],
        "answer": "C"
    },
    {
        "question": "Pre-pruning may use:",
        "options": [
            "Chi-square test",
            "Cross-validation",
            "PCA",
            "Standardization"
        ],
        "answer": "A"
    },
    {
        "question": "Why is post-pruning preferred over pre-pruning?",
        "options": [
            "It is faster",
            "Pre-pruning may stop too early (e.g., on XOR problems)",
            "It avoids EDA",
            "It increases tree depth"
        ],
        "answer": "B"
    },
    {
        "question": "A pure node has:",
        "options": [
            "Maximum entropy",
            "Zero entropy and zero Gini",
            "High variance",
            "Mixed classes"
        ],
        "answer": "B"
    },
    {
        "question": "Which algorithm uses Information Gain with entropy?",
        "options": ["CART", "C4.5", "K-means", "Naïve Bayes"],
        "answer": "B"
    },
    {
        "question": "Decision trees are:",
        "options": [
            "Parametric models",
            "Non-parametric models",
            "Linear models",
            "Probabilistic graphical models"
        ],
        "answer": "B"
    },
    {
        "question": "In regression trees, leaf nodes predict:",
        "options": ["Mode", "Median", "Mean", "Maximum"],
        "answer": "C"
    },
    {
        "question": "Estimating class probabilities in a decision tree involves:",
        "options": [
            "Counting support vectors",
            "Using the ratio of class instances in the leaf node",
            "Applying Laplace smoothing",
            "Calculating Euclidean distance"
        ],
        "answer": "B"
    },
    {
        "question": "Attributes with many values (e.g., ID codes) cause:",
        "options": [
            "Underfitting",
            "Bias in Information Gain (overfitting)",
            "Better generalization",
            "Faster training"
        ],
        "answer": "B"
    },
    {
        "question": "Decision boundaries in decision trees are:",
        "options": [
            "Linear and smooth",
            "Perpendicular to feature axes",
            "Circular",
            "Polynomial"
        ],
        "answer": "B"
    },
    {
        "question": "Greedy training in decision trees means:",
        "options": [
            "Global optimum is guaranteed",
            "Optimal local split is chosen at each step",
            "All splits are evaluated together",
            "Training is skipped"
        ],
        "answer": "B"
    },
    {
        "question": "The CART algorithm for regression minimizes:",
        "options": [
            "Entropy",
            "Gini",
            "Mean Squared Error (MSE)",
            "Information Gain"
        ],
        "answer": "C"
    },
    {
        "question": "Subtree replacement in pruning involves:",
        "options": [
            "Removing a subtree and replacing it with a leaf",
            "Adding new branches",
            "Increasing depth",
            "Standardizing data"
        ],
        "answer": "A"
    },
    {
        "question": "Which metric is maximized when all classes are equally likely?",
        "options": ["Accuracy", "Entropy", "Precision", "F1-score"],
        "answer": "B"
    },
    {
        "question": "Which impurity measure satisfies the multistage property?",
        "options": ["Gini Index", "Variance", "Entropy", "MSE"],
        "answer": "C"
    },
    {
        "question": "Naïve Bayes assumes that features are:",
        "options": [
            "Highly correlated",
            "Conditionally independent given the class",
            "Normally distributed",
            "Linearly separable"
        ],
        "answer": "B"
    },
    {
        "question": "The 'zero-frequency problem' in Naïve Bayes is solved using:",
        "options": ["Standardization", "Laplace smoothing", "PCA", "One-hot encoding"],
        "answer": "B"
    },
    {
        "question": "For numeric attributes, Naïve Bayes typically assumes a:",
        "options": [
            "Uniform distribution",
            "Poisson distribution",
            "Gaussian distribution",
            "Binomial distribution"
        ],
        "answer": "C"
    },
    {
        "question": "The 1R classifier selects the attribute with:",
        "options": [
            "Highest information gain",
            "Lowest classification error",
            "Most unique values",
            "Highest Gini"
        ],
        "answer": "B"
    },
    {
        "question": "In SVM, the optimal decision boundary is called a:",
        "options": ["Decision stump", "Hyperplane", "Cluster centroid", "Probability threshold"],
        "answer": "B"
    },
    {
        "question": "Support vectors are:",
        "options": [
            "All training points",
            "Points closest to the hyperplane",
            "Outliers",
            "Centroids"
        ],
        "answer": "B"
    },
    {
        "question": "Which kernel allows SVM to handle non-linear data?",
        "options": ["Linear", "Identity", "RBF (Radial Basis Function)", "Constant"],
        "answer": "C"
    },
    {
        "question": "Why might Naïve Bayes still perform well despite violating the independence assumption?",
        "options": [
            "It uses deep learning",
            "It only needs to assign the highest probability to the correct class",
            "It ignores probabilities",
            "It uses SVM internally"
        ],
        "answer": "B"
    },
    {
        "question": "Probability densities in Naïve Bayes for numeric attributes:",
        "options": [
            "Must be ≤ 1",
            "Can be > 1",
            "Are always integers",
            "Are replaced by counts"
        ],
        "answer": "B"
    },
    {
        "question": "In 1R, missing values are treated as:",
        "options": [
            "Errors",
            "A separate attribute value",
            "Mean values",
            "Ignored"
        ],
        "answer": "B"
    },
    {
        "question": "Which statement about SVM is TRUE?",
        "options": [
            "It does not require feature scaling",
            "It is sensitive to outliers",
            "It always uses linear kernels",
            "It cannot handle high dimensions"
        ],
        "answer": "B"
    },
    {
        "question": "The main idea of kernel methods in SVM is to:",
        "options": [
            "Reduce training time",
            "Use linear methods on non-linear patterns in transformed space",
            "Replace decision trees",
            "Avoid pre-processing"
        ],
        "answer": "B"
    },
    {
        "question": "In Naïve Bayes, missing attributes during classification are:",
        "options": [
            "Replaced with zero",
            "Omitted from calculation",
            "Dropped from the dataset",
            "Imputed with mean"
        ],
        "answer": "B"
    },
    {
        "question": "SVM is resistant to:",
        "options": [
            "Underfitting",
            "The curse of dimensionality",
            "Small datasets",
            "Linearly separable data"
        ],
        "answer": "B"
    },
    {
        "question": "Which algorithm is described as 'simple but surprisingly effective' by Holte?",
        "options": ["SVM", "1R", "K-means", "PCA"],
        "answer": "B"
    },
    {
        "question": "Laplace smoothing adds:",
        "options": [
            "1 to every attribute-class count",
            "Noise to data",
            "New features",
            "Outliers"
        ],
        "answer": "A"
    },
    {
        "question": "For discretizing numeric attributes in 1R, breakpoints are placed where:",
        "options": [
            "Standard deviation is max",
            "Class changes",
            "Mean is zero",
            "Entropy is max"
        ],
        "answer": "B"
    },
    {
        "question": "SVM optimization aims to:",
        "options": [
            "Minimize classification error",
            "Maximize margin between classes",
            "Minimize tree depth",
            "Maximize likelihood"
        ],
        "answer": "B"
    },
    {
        "question": "Which is NOT a Naïve Bayes assumption?",
        "options": [
            "Feature independence",
            "Equal feature importance",
            "Gaussian distribution for numeric data",
            "Linear decision boundary"
        ],
        "answer": "D"
    },
    {
        "question": "In the weather dataset, P('yes' | evidence) is calculated using:",
        "options": [
            "Euclidean distance",
            "Product of conditional probabilities",
            "Gini index",
            "K-means clustering"
        ],
        "answer": "B"
    },
    {
        "question": "Why is standardization recommended before SVM?",
        "options": [
            "SVM uses distance-based calculations",
            "It improves interpretability",
            "It reduces overfitting",
            "It handles missing values"
        ],
        "answer": "A"
    },
    {
        "question": "The 1R rule for 'Outlook = Sunny' in the weather data predicts:",
        "options": ["Yes", "No", "Maybe", "Unknown"],
        "answer": "B"
    },
    {
        "question": "Naïve Bayes handles redundant attributes poorly because:",
        "options": [
            "It doubles probabilities",
            "Independence assumption is further violated",
            "It crashes",
            "It ignores them"
        ],
        "answer": "B"
    },
    {
        "question": "Kernel functions in SVM:",
        "options": [
            "Must be linear",
            "Can incorporate domain knowledge",
            "Are not modular",
            "Reduce accuracy"
        ],
        "answer": "B"
    },
    {
        "question": "Clustering is a form of:",
        "options": [
            "Supervised learning",
            "Unsupervised learning",
            "Reinforcement learning",
            "Semi-supervised learning"
        ],
        "answer": "B"
    },
    {
        "question": "K-means clustering minimizes:",
        "options": [
            "Entropy",
            "Within-cluster sum of squares",
            "Gini index",
            "Classification error"
        ],
        "answer": "B"
    },
    {
        "question": "Which clustering algorithm can find clusters of arbitrary shape?",
        "options": ["K-means", "K-medoids", "DBSCAN", "Hierarchical (single-link)"],
        "answer": "C"
    },
    {
        "question": "In DBSCAN, a core point has:",
        "options": [
            "No neighbors",
            "At least MinPts points within ε radius",
            "Exactly 2 neighbors",
            "Highest feature value"
        ],
        "answer": "B"
    },
    {
        "question": "The three point types in DBSCAN are:",
        "options": ["Centroid, edge, noise", "Core, border, outlier", "Root, leaf, branch", "Mean, median, mode"],
        "answer": "B"
    },
    {
        "question": "Euclidean distance is used for:",
        "options": [
            "Binary data",
            "Interval-scaled numeric data",
            "Nominal data",
            "Ordinal rankings"
        ],
        "answer": "B"
    },
    {
        "question": "Which clustering method does NOT require specifying k?",
        "options": ["K-means", "K-medoids", "DBSCAN", "All require k"],
        "answer": "C"
    },
    {
        "question": "Agglomerative clustering is:",
        "options": ["Top-down", "Bottom-up", "Density-based", "Model-based"],
        "answer": "B"
    },
    {
        "question": "K-means is weak at handling:",
        "options": [
            "Numeric data",
            "Non-convex clusters",
            "Large datasets",
            "Standardized data"
        ],
        "answer": "B"
    },
    {
        "question": "The Jaccard coefficient is used for:",
        "options": [
            "Continuous variables",
            "Asymmetric binary variables",
            "Ordinal variables",
            "Time-series"
        ],
        "answer": "B"
    },
    {
        "question": "For nominal variables, dissimilarity can be computed as:",
        "options": ["(p – m) / p", "Euclidean distance", "Pearson correlation", "Gini"],
        "answer": "A"
    },
    {
        "question": "Ordinal variables can be mapped to [0,1] using:",
        "options": [
            "z = (rank – 1) / (M – 1)",
            "MinMaxScaler",
            "One-hot encoding",
            "Standardization"
        ],
        "answer": "A"
    },
    {
        "question": "K-means time complexity is approximately:",
        "options": ["O(n)", "O(tkn)", "O(n²)", "O(log n)"],
        "answer": "B"
    },
    {
        "question": "A dendrogram is used in:",
        "options": ["K-means", "DBSCAN", "Hierarchical clustering", "Naïve Bayes"],
        "answer": "C"
    },
    {
        "question": "BIRCH uses a:",
        "options": ["Decision tree", "CF-tree", "Support vector", "Bayesian network"],
        "answer": "B"
    },
    {
        "question": "DBSCAN fails when:",
        "options": [
            "Clusters have varying densities",
            "Data is standardized",
            "MinPts is too low",
            "All points are core"
        ],
        "answer": "A"
    },
    {
        "question": "In clustering, good clusters have:",
        "options": [
            "Low intra-cluster similarity",
            "High inter-cluster similarity",
            "High intra-cluster and low inter-cluster similarity",
            "Random structure"
        ],
        "answer": "C"
    },
    {
        "question": "K-modes is used for:",
        "options": [
            "Numeric data",
            "Categorical data",
            "Mixed data",
            "Time-series"
        ],
        "answer": "B"
    },
    {
        "question": "The mean absolute deviation (MAD) is used to:",
        "options": [
            "Compute Euclidean distance",
            "Standardize interval-scaled data",
            "Encode nominal data",
            "Prune trees"
        ],
        "answer": "B"
    },
    {
        "question": "Which is a requirement for clustering algorithms?",
        "options": [
            "Must handle noise and outliers",
            "Must assume spherical clusters",
            "Must use Euclidean distance",
            "Must specify k"
        ],
        "answer": "A"
    },
    {
        "question": "The Gini Index is used by which decision tree algorithm?",
        "options": ["C4.5", "ID3", "CART", "J48"],
        "answer": "C"
    },
    {
        "question": "Entropy is used by which decision tree algorithm?",
        "options": ["CART", "C4.5", "K-means", "DBSCAN"],
        "answer": "B"
    },
    {
        "question": "What does the IQR stand for?",
        "options": [
            "Inter-Quartile Range",
            "Internal Quality Ratio",
            "Interval Quantile Range",
            "Integrated Query Result"
        ],
        "answer": "A"
    },
    {
        "question": "Which of the following is a filter method for feature selection?",
        "options": [
            "Recursive Feature Elimination",
            "Using correlation coefficients",
            "Genetic algorithms",
            "Forward selection"
        ],
        "answer": "B"
    },
    {
        "question": "Which of the following is a wrapper method for feature selection?",
        "options": [
            "Chi-square test",
            "Information Gain",
            "Forward selection",
            "Variance threshold"
        ],
        "answer": "C"
    },
    {
        "question": "PCA stands for:",
        "options": [
            "Principal Component Analysis",
            "Primary Cluster Algorithm",
            "Probabilistic Classification Approach",
            "Predictive Correlation Analysis"
        ],
        "answer": "A"
    },
    {
        "question": "In Naïve Bayes, the likelihood for a class is converted to a probability by:",
        "options": [
            "Standardization",
            "Normalization (division by total likelihood)",
            "Laplace smoothing",
            "Log transformation"
        ],
        "answer": "B"
    },
    {
        "question": "The main disadvantage of K-means is:",
        "options": [
            "It is slow",
            "It assumes spherical clusters of similar size",
            "It cannot handle numeric data",
            "It requires labeled data"
        ],
        "answer": "B"
    },
    {
        "question": "DBSCAN stands for:",
        "options": [
            "Density-Based Spatial Clustering of Applications with Noise",
            "Data-Based Statistical Clustering Approach for Noise",
            "Decision-Based Supervised Clustering Algorithm",
            "Density-Balanced Spatial Clustering"
        ],
        "answer": "A"
    },
    {
        "question": "Which distance metric is most affected by unscaled features?",
        "options": ["Manhattan distance", "Hamming distance", "Euclidean distance", "Jaccard similarity"],
        "answer": "C"
    },
    {
        "question": "Which of the following is true about decision trees?",
        "options": [
            "They require feature scaling",
            "They are robust to outliers",
            "They cannot handle mixed data types",
            "They always produce balanced trees"
        ],
        "answer": "B"
    },
    {
        "question": "The CRISP-DM phase that evaluates models using business criteria is:",
        "options": ["Data Preparation", "Modelling", "Evaluation", "Deployment"],
        "answer": "C"
    },
    {
        "question": "Which of the following is NOT a data type?",
        "options": ["Nominal", "Ordinal", "Rational", "Interval"],
        "answer": "C"
    },
    {
        "question": "In clustering, the goal is to maximize:",
        "options": [
            "Inter-cluster similarity",
            "Intra-cluster similarity",
            "Total variance",
            "Number of clusters"
        ],
        "answer": "B"
    },
    {
        "question": "The formula for entropy uses:",
        "options": [
            "Natural logarithm",
            "Logarithm base 2",
            "Logarithm base 10",
            "Square root"
        ],
        "answer": "B"
    },
    {
        "question": "In SVM, the margin is:",
        "options": [
            "The number of misclassified points",
            "The distance between support vectors and the hyperplane",
            "The regularization parameter",
            "The classification error"
        ],
        "answer": "B"
    },
    {
        "question": "Which of the following is a strength of Naïve Bayes?",
        "options": [
            "Handles missing values well",
            "Requires large datasets",
            "Sensitive to irrelevant features",
            "Needs feature scaling"
        ],
        "answer": "A"
    },
    {
        "question": "Which of the following is a weakness of 1R?",
        "options": [
            "Too complex",
            "Ignores interactions between attributes",
            "Requires numeric data only",
            "Cannot handle missing values"
        ],
        "answer": "B"
    },
    {
        "question": "In decision trees, the root node is selected based on:",
        "options": [
            "Random selection",
            "Lowest Gini or highest Information Gain",
            "Attribute with most missing values",
            "Alphabetical order"
        ],
        "answer": "B"
    },
    {
        "question": "Which of the following algorithms is non-parametric?",
        "options": [
            "Linear Regression",
            "Decision Tree",
            "Logistic Regression",
            "Naïve Bayes"
        ],
        "answer": "B"
    },
    {
        "question": "The purpose of pruning is to:",
        "options": [
            "Increase tree depth",
            "Reduce overfitting",
            "Add more features",
            "Speed up training"
        ],
        "answer": "B"
    },
    {
        "question": "In EDA, boxplots are used primarily to:",
        "options": [
            "Show correlation",
            "Detect outliers",
            "Encode categorical data",
            "Normalize features"
        ],
        "answer": "B"
    },
    {
        "question": "Which of the following is true about standardization?",
        "options": [
            "Scales data to [0,1]",
            "Is more sensitive to outliers than normalization",
            "Is less sensitive to outliers than normalization",
            "Cannot be used with SVM"
        ],
        "answer": "C"
    },
    {
        "question": "The ID code attribute causes overfitting in decision trees because:",
        "options": [
            "It has high information gain but poor predictive power",
            "It is categorical",
            "It is numeric",
            "It is missing often"
        ],
        "answer": "A"
    },
    {
        "question": "In probabilistic classification, P(class | evidence) is called:",
        "options": [
            "Prior probability",
            "Likelihood",
            "Posterior probability",
            "Marginal probability"
        ],
        "answer": "C"
    },
    {
        "question": "The prior probability P(class) in Naïve Bayes is estimated as:",
        "options": [
            "Number of instances with that class / total instances",
            "Mean of the class",
            "Standard deviation",
            "Gini index"
        ],
        "answer": "A"
    },
    {
        "question": "Which of the following is used to handle nominal data in clustering?",
        "options": [
            "Euclidean distance",
            "Jaccard coefficient",
            "Manhattan distance",
            "Cosine similarity"
        ],
        "answer": "B"
    },
    {
        "question": "In DBSCAN, a border point is:",
        "options": [
            "A core point with low density",
            "A non-core point in the neighborhood of a core point",
            "Always an outlier",
            "A centroid"
        ],
        "answer": "B"
    },
    {
        "question": "Which step in KDD consumes about 60% of the effort?",
        "options": [
            "Data visualization",
            "Data cleaning and pre-processing",
            "Model deployment",
            "Business understanding"
        ],
        "answer": "B"
    },
    {
        "question": "According to the lecture, which is a ratio-scaled data example?",
        "options": [
            "Temperature in Celsius",
            "Calendar year",
            "Distance in kilometers",
            "Letter grade"
        ],
        "answer": "C"
    },
    {
        "question": "How many new binary features are created by one-hot encoding a nominal feature with 5 categories?",
        "options": ["1", "4", "5", "6"],
        "answer": "C"
    },
    {
        "question": "Which clustering algorithm explicitly expects standardized data?",
        "options": ["K-means", "K-medoids", "DBSCAN", "Agglomerative clustering"],
        "answer": "C"
    },
    {
        "question": "What is the main drawback of using ID code as a decision tree attribute?",
        "options": [
            "Increases training time",
            "Causes data fragmentation and overfitting",
            "Cannot be visualized",
            "Requires normalization"
        ],
        "answer": "B"
    },
    {
        "question": "Which impurity measure is the only one that satisfies the multistage property?",
        "options": ["Gini Index", "Variance", "Entropy", "Mean Squared Error"],
        "answer": "C"
    },
    {
        "question": "What is the purpose of Laplace smoothing in Naïve Bayes?",
        "options": [
            "To normalize probabilities",
            "To handle the zero-frequency problem",
            "To reduce dimensionality",
            "To encode categorical variables"
        ],
        "answer": "B"
    },
    {
        "question": "Why might Naïve Bayes work well despite violating independence?",
        "options": [
            "It uses deep learning",
            "It only needs to assign the highest probability to the correct class",
            "It ignores feature values",
            "It averages predictions"
        ],
        "answer": "B"
    },
    {
        "question": "How does 1R treat missing values?",
        "options": [
            "Deletes them",
            "Imputes with mode",
            "Treats as a separate attribute value",
            "Ignores during training"
        ],
        "answer": "C"
    },
    {
        "question": "Which kernel is explicitly mentioned for non-linear SVM?",
        "options": ["Linear", "Polynomial", "RBF", "Sigmoid"],
        "answer": "C"
    },
    {
        "question": "What is the primary optimization goal of SVM?",
        "options": [
            "Minimize classification error",
            "Maximize the margin between classes",
            "Minimize tree depth",
            "Maximize likelihood"
        ],
        "answer": "B"
    },
    {
        "question": "According to Holte, why does 1R perform well?",
        "options": [
            "Uses deep networks",
            "Handles high-dimensional data",
            "Simple rules work well on common datasets",
            "Requires large training sets"
        ],
        "answer": "C"
    },
    {
        "question": "What does the hyperparameter min_samples_leaf control?",
        "options": [
            "Maximum tree depth",
            "Minimum samples to split a node",
            "Minimum samples in a leaf node",
            "Number of features per split"
        ],
        "answer": "C"
    },
    {
        "question": "Which feature selection method builds models to evaluate subsets?",
        "options": ["Filter", "Wrapper", "Embedded", "PCA"],
        "answer": "B"
    },
    {
        "question": "Why is one-hot encoding a disadvantage?",
        "options": [
            "Assumes order",
            "Causes high dimensionality",
            "Only works with numeric data",
            "Requires normalization"
        ],
        "answer": "B"
    },
    {
        "question": "What does BIRCH stand for?",
        "options": [
            "Balanced Iterative Reducing and Clustering using Hierarchies",
            "Binary Iterative Recursive Clustering Heuristic",
            "Bayes-Informed Recursive Clustering Heuristic",
            "Balanced Incremental Recursive Clustering Heuristic"
        ],
        "answer": "A"
    },
    {
        "question": "What is the CF vector in BIRCH composed of?",
        "options": [
            "(Count, Mean, Variance)",
            "(N, LS, SS)",
            "(Min, Median, Max)",
            "(Centroid, Radius, Weight)"
        ],
        "answer": "B"
    },
    {
        "question": "Which distance metric is appropriate for asymmetric binary variables?",
        "options": [
            "Euclidean",
            "Manhattan",
            "Jaccard coefficient",
            "Cosine similarity"
        ],
        "answer": "C"
    },
    {
        "question": "How are ordinal ranks mapped to [0,1]?",
        "options": [
            "z = rank / M",
            "z = (rank - 1) / (M - 1)",
            "z = rank / std",
            "z = (M - rank) / M"
        ],
        "answer": "B"
    },
    {
        "question": "Which is NOT a requirement for good clustering algorithms?",
        "options": [
            "Handle noise and outliers",
            "Discover arbitrary shapes",
            "Assume spherical clusters",
            "Be scalable"
        ],
        "answer": "C"
    },
    {
        "question": "What are support vectors in SVM?",
        "options": [
            "All training instances",
            "Only misclassified points",
            "Points closest to the decision boundary",
            "Class centroids"
        ],
        "answer": "C"
    },
    {
        "question": "What is a disadvantage of PCA?",
        "options": [
            "Increases interpretability",
            "Creates hard-to-interpret linear combinations",
            "Requires labeled data",
            "Is a wrapper method"
        ],
        "answer": "B"
    },
    {
        "question": "What is the time complexity of prediction in a decision tree?",
        "options": [
            "O(n)",
            "O(n log n)",
            "O(log m) where m = training instances",
            "O(k) where k = features"
        ],
        "answer": "C"
    },
    {
        "question": "Which algorithm is described as 'greedy'?",
        "options": [
            "K-means",
            "Naïve Bayes",
            "Decision Tree",
            "DBSCAN"
        ],
        "answer": "C"
    },
    {
        "question": "In Naïve Bayes, what is P(class | evidence) called?",
        "options": ["Prior", "Likelihood", "Posterior", "Marginal"],
        "answer": "C"
    },
    {
        "question": "How is P(class) estimated in Naïve Bayes?",
        "options": [
            "Using Gaussian distribution",
            "As proportion of class instances",
            "By cross-validation",
            "Using Laplace only"
        ],
        "answer": "B"
    },
    {
        "question": "What is a mentioned disadvantage of decision trees?",
        "options": [
            "Require feature scaling",
            "Sensitive to rotation of data",
            "Cannot handle numeric features",
            "Are parametric"
        ],
        "answer": "B"
    },
    {
        "question": "What is the main idea of kernel methods in SVM?",
        "options": [
            "Reduce dimensionality",
            "Use linear methods on non-linear patterns via embedding",
            "Replace decision trees",
            "Handle missing values"
        ],
        "answer": "B"
    },
    {
        "question": "Which is a filter method for feature selection?",
        "options": [
            "Recursive Feature Elimination",
            "Information Gain",
            "Genetic algorithms",
            "Forward selection"
        ],
        "answer": "B"
    },
    {
        "question": "What does the lecture say about zip code?",
        "options": [
            "Always safe to share",
            "Can act as a proxy for race",
            "Required for data mining",
            "Has no predictive power"
        ],
        "answer": "B"
    },
    {
        "question": "Which process model includes a 'Deployment' phase?",
        "options": [
            "KDD only",
            "CRISP-DM only",
            "ASUM only",
            "Both CRISP-DM and ASUM"
        ],
        "answer": "D"
    },
    {
        "question": "What is the main disadvantage of K-means?",
        "options": [
            "Too slow",
            "Assumes spherical, same-size clusters",
            "Cannot handle numeric data",
            "Requires labels"
        ],
        "answer": "B"
    },
    {
        "question": "Why is post-pruning preferred?",
        "options": [
            "Faster than pre-pruning",
            "Pre-pruning may stop too early (e.g., on XOR)",
            "Uses chi-square by default",
            "Guarantees optimal tree"
        ],
        "answer": "B"
    },
    {
        "question": "What defines a core point in DBSCAN?",
        "options": [
            "No neighbors",
            "At least MinPts within ε radius",
            "Is an outlier",
            "Is a centroid"
        ],
        "answer": "B"
    },
    {
        "question": "What is used to standardize interval-scaled data?",
        "options": [
            "One-hot encoding",
            "Min-max scaling",
            "Mean absolute deviation (MAD)",
            "Label encoding"
        ],
        "answer": "C"
    },
    {
        "question": "What is the main goal of feature selection?",
        "options": [
            "Increase model complexity",
            "Improve accuracy, speed, and interpretability",
            "Add more features",
            "Replace pre-processing"
        ],
        "answer": "B"
    },
    {
        "question": "Which algorithm is explicitly NOT part of the DM module?",
        "options": [
            "Supervised learning",
            "Unsupervised learning",
            "Reinforcement learning",
            "Classification"
        ],
        "answer": "C"
    },
    {
        "question": "How is class probability estimated at a leaf node?",
        "options": [
            "Mode of the class",
            "Ratio of class instances in that leaf",
            "Mean of the target",
            "Gini index"
        ],
        "answer": "B"
    },
    {
        "question": "What is a strength of decision trees?",
        "options": [
            "Require feature scaling",
            "Handle both nominal and numeric features",
            "Extrapolate well",
            "Are parametric"
        ],
        "answer": "B"
    },
    {
        "question": "What happens to missing attributes in Naïve Bayes classification?",
        "options": [
            "Replaced with zero",
            "Omitted from calculation",
            "Crash the model",
            "Imputed with mean"
        ],
        "answer": "B"
    },
    {
        "question": "Which distance metric is most affected by unscaled features?",
        "options": ["Manhattan", "Hamming", "Euclidean", "Jaccard"],
        "answer": "C"
    },
    {
        "question": "What is the entropy formula based on?",
        "options": [
            "Natural log",
            "Log base 2",
            "Log base 10",
            "Square root"
        ],
        "answer": "B"
    },
    {
        "question": "Why is standardization recommended for SVM?",
        "options": [
            "SVM uses distance-based calculations",
            "Improves visualization",
            "Reduces overfitting",
            "Handles categorical data"
        ],
        "answer": "A"
    },
    {
        "question": "Which is a wrapper method?",
        "options": [
            "Chi-square test",
            "Information Gain",
            "Forward selection",
            "Variance threshold"
        ],
        "answer": "C"
    },
    {
        "question": "What is the main idea of density-based clustering?",
        "options": [
            "Minimize within-cluster sum of squares",
            "Find dense regions separated by sparse regions",
            "Use hierarchical merging",
            "Assume spherical clusters"
        ],
        "answer": "B"
    },
    {
        "question": "What is a border point in DBSCAN?",
        "options": [
            "Outlier",
            "Core point",
            "Non-core point near a core point",
            "Centroid"
        ],
        "answer": "C"
    },
    {
        "question": "In the weather dataset, what is P('yes')?",
        "options": ["Likelihood", "Prior", "Posterior", "Conditional"],
        "answer": "B"
    },
    {
        "question": "What is a disadvantage of wrapper methods?",
        "options": [
            "They are fast",
            "They are computationally heavy",
            "They ignore feature interactions",
            "They are univariate"
        ],
        "answer": "B"
    },
    {
        "question": "What is the purpose of metadata in data mining?",
        "options": [
            "Store raw data",
            "Restrict search space using background knowledge",
            "Replace EDA",
            "Visualize results"
        ],
        "answer": "B"
    },
    {
        "question": "Which algorithm is used for association rule mining?",
        "options": ["K-means", "Decision Tree", "Apriori", "Naïve Bayes"],
        "answer": "C"
    },
    {
        "question": "What is the main output of regression learning?",
        "options": [
            "Class labels",
            "Association rules",
            "Continuous numeric values",
            "Cluster assignments"
        ],
        "answer": "C"
    },
    {
        "question": "What is a valid application of association rule mining?",
        "options": [
            "Customer segmentation",
            "Market basket analysis",
            "House price prediction",
            "Medical diagnosis"
        ],
        "answer": "B"
    },
    {
        "question": "When are training/test sets created in CRISP-DM?",
        "options": [
            "Business Understanding",
            "Data Understanding",
            "Data Preparation",
            "Modelling"
        ],
        "answer": "C"
    },
    {
        "question": "Which data type allows only equality comparisons?",
        "options": ["Ordinal", "Interval", "Ratio", "Nominal"],
        "answer": "D"
    },
    {
        "question": "What is the purpose of the multistage property?",
        "options": [
            "Allow decisions in several stages",
            "Maximize entropy",
            "Minimize Gini",
            "Speed up training"
        ],
        "answer": "A"
    },
    {
        "question": "Which is true about Gini Index?",
        "options": [
            "Used by C4.5",
            "Slower than entropy",
            "Used by CART",
            "Produces more balanced trees"
        ],
        "answer": "C"
    },
    {
        "question": "What causes overfitting with ID code attributes?",
        "options": [
            "Low information gain",
            "High information gain due to many unique values",
            "Missing values",
            "Categorical encoding"
        ],
        "answer": "B"
    },
    {
        "question": "Which is a non-parametric model?",
        "options": [
            "Linear regression",
            "Logistic regression",
            "Decision tree",
            "Naïve Bayes"
        ],
        "answer": "C"
    },
    {
        "question": "What is the purpose of pruning?",
        "options": [
            "Increase depth",
            "Reduce overfitting",
            "Add features",
            "Speed up training"
        ],
        "answer": "B"
    },
    {
        "question": "What is the primary use of boxplots in EDA?",
        "options": [
            "Show correlation",
            "Detect outliers",
            "Encode categories",
            "Normalize features"
        ],
        "answer": "B"
    },
    {
        "question": "What is a strength of Naïve Bayes?",
        "options": [
            "Handles missing values well",
            "Requires large datasets",
            "Sensitive to irrelevant features",
            "Needs feature scaling"
        ],
        "answer": "A"
    },
    {
        "question": "Which algorithm is simplest but surprisingly effective?",
        "options": ["SVM", "1R", "K-means", "Decision Tree"],
        "answer": "B"
    },
    {
        "question": "What happens in DBSCAN with varying densities?",
        "options": [
            "Works perfectly",
            "Fails to detect all clusters",
            "Creates more clusters",
            "Merges all clusters"
        ],
        "answer": "B"
    },
    {
        "question": "Which is a requirement for clustering algorithms?",
        "options": [
            "Must handle noise and outliers",
            "Must assume convex clusters",
            "Must use Euclidean distance",
            "Must specify k"
        ],
        "answer": "A"
    },
    {
        "question": "What is an advantage of filter methods?",
        "options": [
            "Best accuracy",
            "Computational efficiency and scalability",
            "High interpretability",
            "Handles feature interactions"
        ],
        "answer": "B"
    },
    {
        "question": "Which is used for categorical data in clustering?",
        "options": [
            "Euclidean distance",
            "Jaccard coefficient",
            "Manhattan distance",
            "Cosine similarity"
        ],
        "answer": "B"
    },
    {
        "question": "In Naïve Bayes, what is P(evidence | class) called?",
        "options": ["Prior", "Likelihood", "Posterior", "Evidence"],
        "answer": "B"
    },
    {
        "question": "What is a weakness of 1R?",
        "options": [
            "Too complex",
            "Ignores interactions between attributes",
            "Requires numeric data",
            "Cannot handle missing values"
        ],
        "answer": "B"
    },
    {
        "question": "How is the root node selected in decision trees?",
        "options": [
            "Randomly",
            "Based on lowest Gini or highest Information Gain",
            "Alphabetical order",
            "Feature name length"
        ],
        "answer": "B"
    },
    {
        "question": "Which is true about standardization?",
        "options": [
            "Scales to [0,1]",
            "More sensitive to outliers than normalization",
            "Less sensitive to outliers than normalization",
            "Cannot be used with SVM"
        ],
        "answer": "C"
    },
    {
        "question": "What is the primary goal of clustering?",
        "options": [
            "Predict class labels",
            "Maximize inter-cluster similarity",
            "Maximize intra-cluster and minimize inter-cluster similarity",
            "Reduce dimensionality"
        ],
        "answer": "C"
    },
    {
        "question": "In SVM, what is the margin?",
        "options": [
            "Number of support vectors",
            "Distance between support vectors and hyperplane",
            "Regularization parameter",
            "Classification error"
        ],
        "answer": "B"
    },
    {
        "question": "What is a strength of DBSCAN?",
        "options": [
            "Handles arbitrary-shaped clusters",
            "Requires specifying k",
            "Assumes spherical clusters",
            "Sensitive only to parameters"
        ],
        "answer": "A"
    },
    {
        "question": "What is the main idea of the KDD process?",
        "options": [
            "Replace databases",
            "Extract and transform information into usable knowledge",
            "Visualize data only",
            "Store data efficiently"
        ],
        "answer": "B"
    },
    {
        "question": "Which is a valid data mining task?",
        "options": [
            "Data storage",
            "Pattern discovery",
            "Data deletion",
            "File compression"
        ],
        "answer": "B"
    },
    {
        "question": "What percentage of Americans are identifiable by zip, birth date, and sex?",
        "options": ["50%", "65%", "85%", "95%"],
        "answer": "C"
    },
    {
        "question": "Which is NOT a CRISP-DM phase?",
        "options": [
            "Business Understanding",
            "Data Preparation",
            "Model Interpretation",
            "Deployment"
        ],
        "answer": "C"
    },
    {
        "question": "What is the purpose of feature engineering?",
        "options": [
            "Increase dimensionality",
            "Create informative features for modeling",
            "Delete all original features",
            "Replace clustering"
        ],
        "answer": "B"
    },
    {
        "question": "Which algorithm is robust to outliers?",
        "options": [
            "Linear regression",
            "Decision tree",
            "K-means",
            "Multiple regression"
        ],
        "answer": "B"
    },
    {
        "question": "What distribution is assumed for numeric attributes in Naïve Bayes?",
        "options": ["Uniform", "Poisson", "Gaussian", "Binomial"],
        "answer": "C"
    },
    {
        "question": "What is a disadvantage of K-modes?",
        "options": [
            "Cannot handle numeric data",
            "Assumes Gaussian distribution",
            "Requires feature scaling",
            "Sensitive to initialization"
        ],
        "answer": "A"
    },
    {
        "question": "What is the difference between K-means and K-medoids?",
        "options": [
            "K-means uses actual data points as centers",
            "K-medoids uses actual data points as centers",
            "K-means is for categorical data",
            "K-medoids is faster"
        ],
        "answer": "B"
    },
    {
        "question": "What is a strength of hierarchical clustering?",
        "options": [
            "Scalability",
            "No need to specify k",
            "Handles noise well",
            "Fast computation"
        ],
        "answer": "B"
    },
    {
        "question": "What is the training complexity of decision trees?",
        "options": [
            "O(n)",
            "O(n log n)",
            "O(n × m log m) where n=features, m=instances",
            "O(k)"
        ],
        "answer": "C"
    },
    {
        "question": "What is the MDL principle used for in pruning?",
        "options": [
            "Maximize tree depth",
            "Minimize description length of tree and data",
            "Increase overfitting",
            "Ignore error rates"
        ],
        "answer": "B"
    },
    {
        "question": "What should be attached to data mining results?",
        "options": ["Caveats", "Marketing materials", "Source code", "Passwords"],
        "answer": "A"
    },
    {
        "question": "What is a valid reason for missing values?",
        "options": [
            "Equipment malfunction",
            "Changes in experimental design",
            "Collation of datasets",
            "All of the above"
        ],
        "answer": "D"
    },
    {
        "question": "What is the purpose of dimensionality reduction?",
        "options": [
            "Increase features",
            "Reduce noise and redundancy",
            "Add complexity",
            "Replace pre-processing"
        ],
        "answer": "B"
    },
    {
        "question": "What characterizes non-parametric models?",
        "options": [
            "Fixed number of parameters",
            "Number of parameters grows with data",
            "Always linear",
            "Require feature scaling"
        ],
        "answer": "B"
    },
    {
        "question": "What does RBF stand for in SVM?",
        "options": [
            "Random Binary Features",
            "Radial Basis Function",
            "Recursive Bayesian Filter",
            "Regularized Binary Function"
        ],
        "answer": "B"
    },
    {
        "question": "Which is a step in EDA?",
        "options": [
            "Data cleaning",
            "Statistical analysis",
            "Data visualization",
            "All of the above"
        ],
        "answer": "D"
    },
    {
        "question": "What is the primary measure used in K-means?",
        "options": [
            "Entropy",
            "Within-cluster sum of squares",
            "Gini index",
            "Information gain"
        ],
        "answer": "B"
    },
    {
        "question": "What are decision boundaries like in decision trees?",
        "options": [
            "Smooth curves",
            "Perpendicular to feature axes",
            "Circular",
            "Polynomial"
        ],
        "answer": "B"
    },
    {
        "question": "What can probability densities be in Naïve Bayes?",
        "options": [
            "Only ≤ 1",
            "Greater than 1",
            "Always integers",
            "Always zero"
        ],
        "answer": "B"
    },
    {
        "question": "What is a valid clustering evaluation metric?",
        "options": [
            "Accuracy",
            "Within-cluster similarity",
            "F1-score",
            "Precision"
        ],
        "answer": "B"
    },
    {
        "question": "What is the main idea of the 1R algorithm?",
        "options": [
            "Use all attributes equally",
            "Use the single best attribute for classification",
            "Use deep learning",
            "Use SVM"
        ],
        "answer": "B"
    },
    {
        "question": "What is a disadvantage of deep learning for feature selection?",
        "options": [
            "It is interpretable",
            "It is a black box with unreliable explanations",
            "It is fast",
            "It requires small datasets"
        ],
        "answer": "B"
    },
    {
        "question": "How are class probabilities estimated in decision trees?",
        "options": [
            "Support vectors",
            "Ratio of class instances in the leaf",
            "Laplace smoothing only",
            "Gaussian distribution"
        ],
        "answer": "B"
    },
    {
        "question": "What is a strength of K-means?",
        "options": [
            "Handles arbitrary shapes",
            "Efficient with O(tkn) complexity",
            "Handles noise well",
            "No need to specify k"
        ],
        "answer": "B"
    },
    {
        "question": "What is the chi-square test used for in pre-pruning?",
        "options": [
            "Maximize depth",
            "Test statistical significance of attribute-class association",
            "Increase overfitting",
            "Replace post-pruning"
        ],
        "answer": "B"
    },
    {
        "question": "Which data type does not allow meaningful addition?",
        "options": ["Ratio", "Interval", "Ordinal", "Numeric"],
        "answer": "C"
    },
    {
        "question": "What is density-reachability in DBSCAN?",
        "options": [
            "Symmetric relation",
            "Asymmetric relation",
            "Always transitive",
            "Only for core points"
        ],
        "answer": "B"
    },
    {
        "question": "What is a use of metadata?",
        "options": [
            "Dimensional correctness in expressions",
            "Replace training data",
            "Increase outliers",
            "Delete features"
        ],
        "answer": "A"
    },
    {
        "question": "What is the main output of classification learning?",
        "options": [
            "Numeric predictions",
            "Discrete class labels",
            "Cluster centroids",
            "Association rules"
        ],
        "answer": "B"
    },
    {
        "question": "When should you use standardization over normalization?",
        "options": [
            "When outliers are present",
            "When features are bounded",
            "When using tree-based models",
            "When data is categorical"
        ],
        "answer": "A"
    },
    {
        "question": "What does ASUM stand for?",
        "options": [
            "Analytics Solutions Unified Method",
            "Automated Statistical Unification Model",
            "Advanced Supervised Unified Method",
            "Adaptive Statistical Utility Model"
        ],
        "answer": "A"
    },
    {
        "question": "What characterizes interval data?",
        "options": [
            "Has a true zero",
            "Allows ratio comparisons",
            "Has equal units but no true zero",
            "Only allows equality tests"
        ],
        "answer": "C"
    },
    {
        "question": "What is subtree replacement in pruning?",
        "options": [
            "Add new branches",
            "Replace a subtree with a leaf node",
            "Increase depth",
            "Standardize data"
        ],
        "answer": "B"
    },
    {
        "question": "What is a valid application of clustering?",
        "options": [
            "Fraud detection",
            "Customer segmentation",
            "Weather forecasting",
            "All of the above"
        ],
        "answer": "D"
    }
]

# For demonstration, we'll duplicate to reach ~30 questions.
# In practice, include all 120 using the same format.
while len(QUESTIONS) < 230:
    QUESTIONS.append(QUESTIONS[len(QUESTIONS) % 30])

# ===========================================
# STREAMLIT APP
# ===========================================
def main():
    st.set_page_config(page_title="Data Mining & ML Quiz", layout="centered")
    st.title("Data Mining & Machine Learning Quiz")
    st.markdown("Based on Lecture Notes – Test Your Knowledge!")

    total = len(QUESTIONS)
    
    # Initialize session state
    if "current" not in st.session_state:
        st.session_state.current = 0
        st.session_state.score = 0
        st.session_state.submitted = False

    idx = st.session_state.current
    q = QUESTIONS[idx]

    # Progress bar
    st.progress((idx + 1) / total)
    st.subheader(f"Question {idx + 1} of {total}")
    st.write(f"**{q['question']}**")

    # Radio buttons with full option text
    user_ans = st.radio(
        "Select your answer:",
        [f"A. {q['options'][0]}", 
         f"B. {q['options'][1]}", 
         f"C. {q['options'][2]}", 
         f"D. {q['options'][3]}"],
        index=None,
        key=f"q{idx}"
    )

    # Submit button
    if st.button("✅ Submit"):
        if user_ans is None:
            st.warning("Please select an answer.")
        else:
            st.session_state.submitted = True
            selected_letter = user_ans[0]  # Extract 'A', 'B', etc.
            correct_letter = q["answer"]
            correct_text = q["options"][ord(correct_letter) - ord("A")]
            
            if selected_letter == correct_letter:
                st.success("✅ Correct!")
                st.session_state.score += 1
            else:
                st.error(f"❌ Incorrect. The correct answer is **{correct_letter}. {correct_text}**")

    # Next / Finish logic
    if st.session_state.submitted:
        if idx < total - 1:
            if st.button("➡️ Next Question"):
                st.session_state.current += 1
                st.session_state.submitted = False
                st.rerun()
        else:
            st.divider()
            final_score = st.session_state.score
            st.title("🎉 Quiz Completed!")
            st.subheader(f"Your Final Score: **{final_score} / {total}**")
            percentage = (final_score / total) * 100
            st.metric("Accuracy", f"{percentage:.1f}%")
            if percentage >= 80:
                st.balloons()
                st.success("Excellent work! 🌟")
            elif percentage >= 60:
                st.info("Good job! Keep studying.")
            else:
                st.warning("Review the material and try again! 💪")

if __name__ == "__main__":
    main()


