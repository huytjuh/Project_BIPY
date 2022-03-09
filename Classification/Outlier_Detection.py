### OUTLIER DETECTION ###

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def OneClassSVM_outlier(df, contamination=0.1, dict_param={}, random_seed=1234):    
    OCSVM = OneClassSVM(nu=contamination, *dict_param).fit(df)
    df_output = df.copy()
    df_output['Outlier'] = [1 if x == -1 else 0 for x in OCSVM.predict(df)]
    return df_output

def IsolationForest_outlier(df, n_estimators=1000, contamination=0.1, dict_param={}, random_seed=1234):
    IF = IsolationForest(contamination=contamination, *dict_param).fit(df)
    df_output = df.copy()
    df_output['Outlier'] = [1 if x == -1 else 0 for x in IF.predict(df)]
    return

def FSRMCD_MAC_outlier(df):
    output = get_fsrmcdmac_outliers(df)
    return output
