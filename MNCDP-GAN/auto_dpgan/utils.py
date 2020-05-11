import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Feature type map of all possible features for the dataset
features_types = {'Power': 'categorical',
                  'Brand': 'categorical',
                  'Gas': 'categorical binary',
                  'Region': 'categorical',
                  'DriverAge': 'positive int',
                  'DriverAge_bin': 'categorical',
                  'CarAge': 'positive int',
                  'CarAge_bin': 'categorical',
                  'Density': 'positive float',
                  'Density_cat': 'categorical',
                  'Log(Density)_bin': 'categorical',
                  'Exposure': 'positive float',
                  'Exposure_cat': 'categorical',
                  'Exposure_bin': 'categorical',
                  'ClaimNb': 'positive int',
                  'ClaimNb_cat': 'categorical',
                  }

# Features of the MTPL data set for which the type is set
mtpl_features = ["Power", "Brand", "Gas", "Region"]


class Processor:
    def __init__(self, datatypes):
        self.datatypes = datatypes
        self.cutoffs = []
        self.preprocessors = []

    def fit(self, matrix):
        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:, i].reshape(-1, 1)

            if 'categorical' in datatype:
                # Encoding of C categories to C one-hot vectors -> GOOD
                preprocessor = OneHotEncoder(sparse=False)
            else:
                preprocessor = MinMaxScaler()

            preprocessed_col = preprocessor.fit_transform(preprocessed_col)
            self.cutoffs.append(preprocessed_col.shape[1])
            self.preprocessors.append(preprocessor)

    def transform(self, matrix):
        preprocessed_cols = []

        for i, (_, _) in enumerate(self.datatypes):
            preprocessed_col = matrix[:, i].reshape(-1, 1)
            preprocessed_col = self.preprocessors[i].transform(
                preprocessed_col)
            preprocessed_cols.append(preprocessed_col)

        return np.concatenate(preprocessed_cols, axis=1)

    def fit_transform(self, matrix):
        self.fit(matrix)
        return self.transform(matrix)

    def inverse_transform(self, matrix):
        postprocessed_cols = []

        j = 0
        for i, (column, datatype) in enumerate(self.datatypes):
            postprocessed_col = self.preprocessors[i].inverse_transform(
                matrix[:, j:j + self.cutoffs[i]])

            if 'categorical' in datatype:
                postprocessed_col = postprocessed_col.reshape(-1, 1)
            else:
                if 'positive' in datatype:
                    postprocessed_col = postprocessed_col.clip(min=0)

                if 'int' in datatype:
                    postprocessed_col = postprocessed_col.round()

            postprocessed_cols.append(postprocessed_col)

            j += self.cutoffs[i]

        return np.concatenate(postprocessed_cols, axis=1)


def prepare_data(data_path, chosen_features, is_eval=False):
    """Load and process the data according to the chosen features.

    @param data_path: (str) Path to the preprocessed data stored as csv
    @param chosen_features: List of the features to keep in the data
    @param is_eval: (bool) If True, the function will also return the
    fitted processor, the pandas DataFrame of the chosen features and map
    of the names of the categories for each feature (as a dictionary)
    @return: The processed data as a numpy array and the map of the type of
    each feature as a List of (name, type)
    """
    # Load data (shape is indicated in comments)
    df = pd.read_csv(data_path)  # (N x 15)
    # Keep only the chosen features
    df = df[chosen_features]  # (N x 9)

    # Build a map of the type of each feature present in the dataframe
    # Also build a map of the labels of the categories for each feature
    feature_type_map = []  # Each element is a tuple of (name, type)
    cat_names_map = {}  # To be used for plotting the univariate distributions
    for feature in df.columns:
        feature_type_map.append((feature, features_types[feature]))
        cat_names_map[feature] = (df[feature]
                                  .astype("category").cat.categories)

    # Transform categorical features to integers
    for column, datatype in feature_type_map:
        if 'categorical' in datatype:
            df[column] = df[column].astype('category').cat.codes
    data = np.array(df)  # Original data (N x 9) as numeric values only

    # Process data (normalization and one-hot encoding)
    data_processor = Processor(feature_type_map)
    data_processed = (data_processor
                      .fit_transform(data)
                      .astype("float32"))  # (N x 30+)

    if is_eval:
        return (data_processed, feature_type_map,
                data_processor, df, cat_names_map)
    else:
        return data_processed, feature_type_map


def get_processed_feature_names(feature_type_map, df):
    """Make the list of the features corresponding to the processed columns.

    @param feature_type_map: List of (feature name, feature type)
    @param df: pandas DataFrame of the data
    @return: List of the name of each feature in the processed data
    """
    f_names = []  # Categorical features appear more than once
    for f_name, f_type in feature_type_map:
        if "categorical" in f_type:
            # One Hot Encoder makes C vectors where C is the number of cat
            f_names += ([f_name] * df[f_name].nunique())
        else:
            f_names += [f_name]

    return f_names
