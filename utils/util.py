### UTILS ###

from sklearn.impute import KNNImputer

def one_hot_encode(data, drop_columns=[]):
    _input = data.copy()
    output = _input.drop(columns=[x for x in _input.columns if x in drop_columns])
    cat_features = output.select_dtypes(include=['object']).columns.tolist()
    encoded_columns = pd.get_dummies(_input[cat_features], drop_first=True)
    print('Number of categorical columns one-hot-encoded: {:.0f}'.format(len(cat_features)))

    output = pd.concat([output.drop(cat_features, axis=1), encoded_columns], axis=1)
    return output

def impute(data, model='KNN', dict_param={'n_neighbors': 5}, NA=True, outlier=None):
    _input = data.copy()
    model_impute = KNNImputer(**dict_param)
    
    if outlier:
        _input.loc[_input['Outlier']==1, [x for x in _input.drop(columns=['Outlier']).select_dtypes(include=['number']).columns]] = np.nan
    output = pd.DataFrame(model_impute.fit_transform(_input.select_dtypes(include=['number'])), columns=_input.select_dtypes(include=['number']).columns)
    output = pd.concat([_input.select_dtypes(exclude=['number']), output], axis=1)
    return output

def knn_classifier(data, model='KNN'):

    return
