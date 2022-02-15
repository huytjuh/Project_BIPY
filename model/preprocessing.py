import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV


class preprocess():
    
    def convert_date_format(data, date_column, date_format, *args, **kwargs):
        _input = data.copy()
        output = _input
        output[date_column] = output[date_column].apply(lambda x: dt.datetime.strptime(x, date_format).strftime('%d-%m-%Y'))
        return output
    
    def one_hot_encode(data, drop_columns=[]):
        _input = data.copy()
        output = _input.drop(columns=[x for x in _input.columns if x in drop_columns])
        cat_features = output.select_dtypes(include=['object']).columns.tolist()
        encoded_columns = pd.get_dummies(_input[cat_features], drop_first=True)
        print('Number of categorical columns one-hot-encoded: {:.0f}'.format(len(cat_features)))
        
        output = pd.concat([output.drop(cat_features, axis=1), encoded_columns], axis=1)
        return output

    def outlier_detection(data, model='IsolationForest', dict_param={'n_estimators': [50, 100, 250, 500], 'contamination': [0.05, 0.1]}):
        _input = data.copy()
        def scorer_f(estimator, X):
            return np.mean(estimator.score_samples(X))
        model_outlier = GridSearchCV(IsolationForest(random_state=1234), param_grid=dict_param, scoring=scorer_f).fit(_input)
        _input['Outlier'] = [0 if x==1 else 1 for x in model_outlier.predict(_input)]
        output = _input
        return output
    
    def impute(data, model='KNN', dict_param={'n_neighbors': 5}, NA=True, outlier=None):
        _input = data.copy()
        model_impute = KNNImputer(**dict_param)
        
        if outlier:
            _input.loc[_input['Outlier']==1, [x for x in _input.drop(columns=['Outlier']).select_dtypes(include=['number']).columns]] = np.nan
        output = pd.DataFrame(model_impute.fit_transform(_input.select_dtypes(include=['number'])), columns=_input.select_dtypes(include=['number']).columns)
        output = pd.concat([_input.select_dtypes(exclude=['number']), output], axis=1)
        return output
