### CLASSIFICATION ###

import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score, log_loss, ConfusionMatrixDisplay, RocCurveDisplay

!pip install scikit-optimize
from skopt import BayesSearchCV

def one_hot_encode(data, drop_columns=[]):
    _input = data.copy()
    output = _input.drop(columns=[x for x in _input.columns if x in drop_columns])
    cat_features = output.select_dtypes(include=['object']).columns.tolist()
    encoded_columns = pd.get_dummies(_input[cat_features], drop_first=True)
    print('Number of categorical columns one-hot-encoded: {:.0f}'.format(len(cat_features)))

    output = pd.concat([output.drop(cat_features, axis=1), encoded_columns], axis=1)
    return output

def logit_classifier(X_train, X_test, y_train, y_test, weight=0.5, random_seed=1234):
    logit = sm.Logit(y_train, X_train).fit()
    logit_preds = [1 if x>=weight else 0 for x in logit.predict(X_test)]
    
    dict_output = {'Model': 'Logit', 'Preds': logit_preds, 'Summary': logit.summary}
    return dict_output

def DT_classifier(X_train, X_test, y_train, y_test, cv, dict_param={'criterion': ['gini', 'entropy'], 'max_depth': [None]}, n_iter=50, random_seed=1234):
    DT = BayesSearchCV(DecisionTreeClassifier(), dict_param, cv=cv, n_iter=n_iter, random_state=random_seed).fit(X_train, y_train)
    DT_preds = DT.predict(X_test)

    dict_output = {'Model': 'Decision Tree', 'Preds': DT_preds, 'Estimator': DT.best_estimator_}
    return dict_output    

def RF_classifier(X_train, X_test, y_train, y_test, cv, dict_param={'criterion': ['gini', 'entropy'], 'n_estimators': [100, 200, 300, 400], 'max_depth': list(range(2,10))}, n_iter=50, random_seed=1234):
    RF = BayesSearchCV(RandomForestClassifier(), dict_param, cv=cv, n_iter=n_iter, random_state=random_seed).fit(X_train, y_train)
    RF_preds = RF.predict(X_test)

    dict_output = {'Model': 'Random Forest', 'Preds': RF_preds, 'Estimator': RF.best_estimator_}
    return dict_output    

def evaluate_classifier(list_models, y_test, beta=1, visualize=False):
    if not isinstance(list_models, list):
        list_models = [list_models]

    #y_test = y_test.reset_index(drop=True)
    df_output = pd.DataFrame()
    dict_graph = {'Confusion Matrix': {}, 'ROC': {}}
    for result_model in list_models:
        temp = {'Model': result_model['Model'], 
                'Precision': precision_score(y_test, result_model['Preds']),
                'Recall': recall_score(y_test, result_model['Preds']),
                'F{}'.format(beta): fbeta_score(y_test, result_model['Preds'], beta=beta),
                'Accuracy': accuracy_score(y_test, result_model['Preds']),
                'Log loss': log_loss(y_test, result_model['Preds'])}
        
        if visualize==True:
            dict_graph['Confusion Matrix'][result_model['Model']] = ConfusionMatrixDisplay.from_predictions(y_test, result_model['Preds'])
            dict_graph['ROC'][result_model['Model']] = RocCurveDisplay.from_predictions(y_test, result_model['Preds'])

        df_output = df_output.append(temp, ignore_index=True).round(3)

    if visualize==False:
        return df_output
    else:
        return df_output, dict_graph

