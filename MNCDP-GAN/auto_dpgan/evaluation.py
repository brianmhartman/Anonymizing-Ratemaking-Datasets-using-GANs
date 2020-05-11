from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, \
    mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Setup
sns.set()
rf_params = {"n_estimators": 100}  # For Random Forest models


def isolate_column(df, column):
    """Splits a given dataframe into a column and the rest of the dataframe."""
    return df.drop(columns=[column]).values, df[column].values


def get_prediction_score(x_train, y_train, x_test, y_test, model, score):
    """Trains a model on training data and returns the quality of the
    predictions given test data and expected output.
    """
    try:
        # Train model on synthetic data, see how well it predicts test data
        clf = model()
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)

    except ValueError:
        # Only one label present, so simply predict all examples as that label
        prediction = np.full(y_test.shape, y_train[0])

    return score(y_test, prediction)


def feature_prediction_evaluation(
        train, test, synthetic,
        model=lambda: LogisticRegression(solver='lbfgs'),
        score=lambda y_true, y_pred: f1_score(y_true, y_pred)):
    """Runs the evaluation method given real and fake data.

    real: pd.DataFrame
    synthetic: pd.DataFrame, with columns identical to real
    Model: func, takes no arguments and returns object with fit and predict
    functions implemented, e.g. sklearn.linear_model.LogisticRegression
    score: func, takes in a ground truth and a prediction and returns the
    score of the prediction
    """
    if set(train.columns) != set(synthetic.columns):
        raise Exception('Columns of given datasets are not identical.')

    real_clf_scores, synthetic_clf_scores = [], []
    for i, column in enumerate(train.columns):
        x_train_real, y_train_real = isolate_column(train, column)
        x_train_synthetic, y_train_synthetic = isolate_column(synthetic,
                                                              column)
        x_test, y_test = isolate_column(test, column)

        real_score = get_prediction_score(x_train_real, y_train_real,
                                          x_test, y_test, model, score)
        syn_score = get_prediction_score(x_train_synthetic, y_train_synthetic,
                                         x_test, y_test, model, score)

        real_clf_scores.append(real_score)
        synthetic_clf_scores.append(syn_score)
        print(i, real_score, syn_score)

    plt.scatter(real_clf_scores, synthetic_clf_scores, s=3, c='red')
    plt.title('Synthetic Data Evaluation Metric')
    plt.xlabel('Real Data')
    plt.ylabel('Synthetic Data')
    ax = plt.savefig('out.png')
    pd.DataFrame(data={'real': real_clf_scores,
                       'synthetic': synthetic_clf_scores}).to_csv('out.csv')

    return (sum(map(lambda pair: (pair[0] - pair[1]) ** 2,
                    zip(real_clf_scores, synthetic_clf_scores))),
            ax)


def pca_evaluation(real, synthetic):
    pca = PCA(n_components=2)
    pca.fit(real.values)

    real_projection = pca.transform(real.values)
    synthetic_projection = pca.transform(synthetic.values)

    ax = pd.DataFrame(data=real_projection).plot(x=0, y=1, c='red',
                                                 kind='scatter', s=0.5)
    ax = pd.DataFrame(data=synthetic_projection).plot(x=0, y=1, c='blue',
                                                      kind='scatter', s=0.5,
                                                      ax=ax)
    plt.savefig('out.png')

    return ax


def plot_categorical(real, generated, name_of_feature, labels_of_categories,
                     save_name):
    """Plot and save the Barplot of a categorical feature for the real and
    the generated datasets.

    @param real: Pandas Series of the real values for a feature
    @param generated: Pandas Series of the generated values for a feature
    @param name_of_feature: (str) Name of the feature
    @param labels_of_categories: (list-like) Unique labels of the categories
    sorted ascending
    @param save_name: (str) Path of the file to save the plot to
    """
    # Compute relative counts
    real_counts = real.value_counts().sort_index()  # Absolute counts
    real_counts = real_counts / real_counts.sum()  # Convert to relative
    gen_counts = generated.value_counts().sort_index()
    gen_counts = gen_counts / gen_counts.sum()

    # Create DataFrame of the counts
    count_df = pd.DataFrame({"Category": labels_of_categories,
                             "Real": real_counts,
                             "Generated": gen_counts})

    # Unpivot the DataFrame from wide to long format
    melted_count_df = pd.melt(count_df, id_vars="Category")
    melted_count_df = melted_count_df.rename(columns={"variable": "Dataset",
                                                      "value": "Count"})

    # Barplot
    # Adjust size with number of categories
    plt.figure(figsize=(12, 2 + len(labels_of_categories) // 2))
    sns.barplot(y="Category", x="Count", hue="Dataset", data=melted_count_df,
                orient="h")
    plt.ylabel(name_of_feature)
    plt.xlabel("Relative count")
    plt.title(
        f"Relative counts of the feature '{name_of_feature}' "
        f"for the real and generated datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_numeric(real, generated, name_of_feature, save_name):
    """Plot and save the Histogram of a numeric or ordinal feature for
    the real and the generated datasets.

    @param real: Pandas Series of the real values for a feature
    @param generated: Pandas Series of the generated values for a feature
    @param name_of_feature: (str) Name of the feature
    @param save_name: (str) Path of the file to save the plot to
    """
    plt.figure(figsize=(12, 6))
    n_bins = real.nunique()  # number of cat in the generated is usually higher
    if n_bins > 100:  # Looks bad with too many bins
        n_bins = 100

    plt.hist(real,
             weights=np.ones_like(real) / len(real),  # Relative count
             bins=n_bins,
             label="Real")

    plt.hist(generated,
             weights=np.ones_like(generated) / len(generated),
             bins=n_bins,
             alpha=0.5,
             label="Generated")

    plt.ylabel("Relative count")
    plt.xlabel(name_of_feature)
    plt.title(
        f"Relative counts of the feature '{name_of_feature}' "
        f"for the real and generated datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_confusion_matrix(truth, pred, save_name):
    """Plot a confusion matrix of the ground truth and the predictions.

    @param truth: (ndarray) Ground truth target values as integers.
    @param pred: (ndarray) Estimated targets as returned by a classifier.
    @param save_name: (str) Path of the file to save the plot to
    """
    title = "real" if "real" in save_name else "generated"
    labels = np.unique(truth)
    cm = confusion_matrix(truth, pred, labels=labels, normalize="all")

    plt.figure()
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="RdBu_r",
                center=0, linewidths=10, square=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for the {title} data')
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight")
    # plt.show()
    plt.close()


def rf_classification(x_train, y_train, x_test, y_test, feature_names,
                      save_name, cm_as_plot=False):
    """Train and test a Random Forest classifier on the given data.

    @param x_train: (ndarray) Features of the train set
    @param y_train: (ndarray) Target of the train set
    @param x_test: (ndarray) Features of the test set
    @param y_test: (ndarray) Target of the test set
    @param feature_names: (list-like) The corresponding names of the features
    with the columns of the train sets
    @param save_name: (str) Path of the file to save the plot to
    @param cm_as_plot: (bool) If True, will plot and save the confusion matrix
    @return: (str) The results as a nice-looking string
    """
    # Labels of all possible classes
    classes = np.unique(np.concatenate([y_train, y_test]))

    # Fit and predict
    clf = RandomForestClassifier(rf_params["n_estimators"])
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    pred_probs = clf.predict_proba(x_test)  # For multiclass ROC AUC

    # Compute confusion matrix and plot it if desired
    cm = confusion_matrix(y_test, prediction)
    if cm_as_plot:
        plot_confusion_matrix(y_test, prediction, save_name)

    # If the generated data doesn't contain one of the classes, the ROC-AUC
    # score function will raise an error. In that case, we need to adjust
    # the predicted probabilities to have the same dimension as the total
    # number of classes. We do this by putting the value 0 for these missing
    # classes.
    if len(classes) != len(clf.classes_):
        for class_idx in classes:
            if class_idx not in clf.classes_:
                pred_probs = np.insert(pred_probs, int(class_idx), 0, axis=1)

    # Compute multiclass ROC AUC score (insensitive to class imbalance)
    roc_auc = roc_auc_score(y_test, pred_probs,
                            multi_class="ovo", labels=classes)
    # Because of the class imbalance, the least frequent ones might not be in
    # the ground truth. Hence, we can't use "one-vs-rest" and need "one-vs-one"

    # Sort and format the importance scores given to features
    f_importance = feature_by_importance(clf.feature_importances_,
                                         feature_names)

    # Format the results as a nice string
    results_message = clf_results(f_importance, roc_auc, cm)

    return results_message


def rf_regression(x_train, y_train, x_test, y_test, feature_names):
    """Train and test a Random Forest regressor on the given data.

    @param x_train: (ndarray) Features of the train set
    @param y_train: (ndarray) Target of the train set
    @param x_test: (ndarray) Features of the test set
    @param y_test: (ndarray) Target of the test set
    @param feature_names: (list-like) The corresponding names of the features
    with the columns of the train sets
    @return: (str) The results as a nice-looking string
    """
    # Fit and predict
    clf = RandomForestRegressor(rf_params["n_estimators"])
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)

    # Test the model with MSE and R^2
    mse = mean_squared_error(y_test, prediction)
    r2 = clf.score(x_test, y_test)

    # Sort and format the importance scores given to features
    f_importance = feature_by_importance(clf.feature_importances_,
                                         feature_names)

    # Format the results as a nice string
    results_message = reg_results(f_importance, mse, r2)

    return results_message


def feature_by_importance(feature_importance, feature_names):
    """Sort and format the importance scores given to features by a classifier.

    @param feature_importance: (ndarray) Importance score given to each
    feature by the classifier
    @param feature_names: (list-like) The name of each feature in the same
    order as the 'feature_importance'
    @return: pandas Series of the sorted scores with the feature names as
    the index
    """
    # Get the importance of the processed features
    f_importance = pd.Series(feature_importance, index=feature_names)
    # Sum the importance for the binarized features
    f_importance = f_importance.groupby(level=0).sum()  # level 0 is the index
    # Sort each feature by its importance
    f_importance = f_importance.sort_values(ascending=False)

    return f_importance


def clf_results(feature_importance, roc_auc, cm):
    """Format the results of a classifier as a nice-looking string.

    @param feature_importance: (pandas Series) Sorted importance score given to
    each feature by the classifier with the name of the feature as the index
    @param roc_auc: (float) ROC-AUC score of the classifier
    @param cm: (ndarray) Confusion matrix of the classifier
    @return: The results as a nice-looking string
    """
    # Build the results string
    result_message = f"ROC-AUC multiclass score: {roc_auc:.4f}\n"
    result_message += f"Confusion matrix\n" \
                      f"(row = true, col = predicted):\n{cm}\n"
    result_message += "Importance of each feature:\n"
    for name, score in feature_importance.iteritems():
        result_message += f"\t{name:<20} {score:.4f}\n"

    return result_message


def reg_results(feature_importance, mse, r2):
    """Format the results of a regressor as a nice-looking string.

    @param feature_importance: (pandas Series) Sorted importance score given to
    each feature by the classifier with the name of the feature as the index
    @param mse: (float) Mean Squared Error of the regressor
    @param r2: (float) R^2 coefficient of the regressor
    @return: The results as a nice-looking string
    """
    # Build the results string
    result_message = f"Mean Squared Error: {mse:.4f}\n"
    result_message += f"R^2 coefficient: {r2:.4f}\n"
    result_message += "Importance of each feature:\n"
    for name, score in feature_importance.iteritems():
        result_message += f"\t{name:<20} {score:.4f}\n"

    return result_message
