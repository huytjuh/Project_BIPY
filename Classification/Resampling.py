### CLASS IMBALANCE ###

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.utils import resample

def naive_resample(df, target, random_seed=1234):
    X, y = RandomOverSampler(random_state=random_seed).fit_resample(df.drop(columns=target), df[[target]])
    df_output = pd.concat([X, y],axis=1)
    return df_output

def SNOTE_resample(df, target, method='naive', random_seed=1234):
    list_SNOTE = {'naive': SMOTE(random_state=random_seed), 'NC': SMOTENC([df.columns.get_loc(col) for col in df.select_dtypes(include=['object']).columns], random_state=random_seed),
                  'borderline': BorderlineSMOTE(random_state=random_seed, kind='borderline-1'), 'SVM': SVMSMOTE(random_state=random_seed)}
    print(list_SNOTE[method])
    X, y = list_SNOTE[method].fit_resample(df.drop(columns=target), df[[target]])
    df_output = pd.concat([X, y], axis=1)
    return df_output

def ADASYN_resample(df, target, random_seed=1234):
    X, y = ADASYN(random_state=random_seed).fit_resample(df.drop(columns=target), df[[target]])
    df_output = pd.concat([X, y], axis=1)
    return df_output
