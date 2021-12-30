"""DO NOT DELETE ANY PART OF CODE
We will run only the evaluation function.

Do not put anything outside of the functions, it will take time in evaluation.
You will have to create another code file to run the necessary code.
"""

# import statements
# !pip install category_encoders
# !pip install scikit-learn-extra

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder

import pickle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def mapped_labels(labels, sorted_actual_labels):
    sorted_pred_labels = pd.Series(labels).value_counts().sort_values(ascending=False).index.to_numpy()
    mapping_dict = dict(zip(sorted_pred_labels, sorted_actual_labels))

    return pd.Series(labels).replace(mapping_dict)


def train():
    covtype_df = pd.read_csv('covtype_train.csv', skipinitialspace=True)

    """### Preprocessing"""

    skewed_columns = ['Hillshade_9am', 'Hillshade_Noon', 'Elevation']
    covtype_df.drop(columns=skewed_columns, inplace=True)

    encoding_scheme = {'Aspect': {'aspect_low': 0, 'aspect_medium': 1, 'aspect_high': 2, 'aspect_ultra': 3},
                       'Slope': {'slope_low': 0, 'slope_medium': 1, 'slope_high': 2, 'slope_ultra': 3},
                       'Horizontal_Distance_To_Fire_Points': {'low': 0, 'mid': 1, 'high': 2}
                       }

    categorical_col = ['Aspect', 'Slope', 'Horizontal_Distance_To_Fire_Points']

    for column in categorical_col:
        encoder = OrdinalEncoder(cols=column, return_df=True,
                                 mapping=[{'col': column, 'mapping': encoding_scheme[column]}])
        covtype_df[column] = encoder.fit_transform(covtype_df[column])
        covtype_df[column] = covtype_df[column].astype(np.int64)

    target_df = pd.DataFrame(covtype_df['target'], columns=['target'])
    covtype_df.drop(['target'], axis=1, inplace=True)


    components = 2
    pca = PCA(n_components=components)
    pca_data = pca.fit_transform(covtype_df)

    pca_columns = []
    for (item1, item2) in zip(['Feature '] * components, np.arange(1, components + 1, 1)):
        pca_columns.append(item1 + str(item2))

    pca_covtype_df = pd.DataFrame(pca_data, columns=pca_columns)

    # freq based sorted out the cluster labels
    sorted_actual_labels = target_df['target'].value_counts().sort_values(ascending=False).index.to_numpy()

    gmm = GaussianMixture(n_components=7, covariance_type='full', random_state=42)

    gmm.fit(pca_covtype_df)

    # save the model to disk
    pickle.dump(gmm, open("q2.pkl", "wb"))

    return sorted_actual_labels


def predict(test_set):
    # find and load your best model
    # Do all preprocessings inside this function only.
    # predict on the test set provided
    '''
    'test_set' is a csv path "test.csv", You need to read the csv and predict using your model.
    '''

    sorted_actual_labels = train()

    # Read csv file
    test_df = pd.read_csv(test_set)

    # Droping some Columns
    test_df.drop(columns=['Hillshade_9am', 'Hillshade_Noon', 'Elevation'], inplace=True)

    # converting string values to int
    encoding_scheme = {'Aspect': {'aspect_low': 0, 'aspect_medium': 1, 'aspect_high': 2, 'aspect_ultra': 3},
                       'Slope': {'slope_low': 0, 'slope_medium': 1, 'slope_high': 2, 'slope_ultra': 3},
                       'Horizontal_Distance_To_Fire_Points': {'low': 0, 'mid': 1, 'high': 2}
                       }

    categorical_col = ['Aspect', 'Slope', 'Horizontal_Distance_To_Fire_Points']

    for column in categorical_col:
        encoder = OrdinalEncoder(cols=column, return_df=True,
                                 mapping=[{'col': column, 'mapping': encoding_scheme[column]}])
        test_df[column] = encoder.fit_transform(test_df[column])
        test_df[column] = test_df[column].astype(np.int64)

    # PCA Encoding
    components = 2
    pca = PCA(n_components=components)
    pca_data = pca.fit_transform(test_df)

    pca_columns = []
    for (item1, item2) in zip(['Feature '] * components, np.arange(1, components + 1, 1)):
        pca_columns.append(item1 + str(item2))

    pca_covtype_df = pd.DataFrame(pca_data, columns=pca_columns)

    gmm = pickle.load(open("q2.pkl", 'rb'))
    gmm_labels = gmm.predict(pca_covtype_df)
    gmm_labels = mapped_labels(gmm_labels, sorted_actual_labels)

    '''
    prediction is a 1D 'list' of output labels. just a single python list.
    '''
    return gmm_labels.tolist()


ans=predict('covtype_train.csv')
print(set(ans))