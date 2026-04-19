import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


def get_df_grasp(df):
    
    t = df.nunique(axis=0, dropna=True).reset_index() # unique values excluding NaN
    t.columns = ['feature', 'unique_values_cnt']
    t['data_type'] = df.dtypes.reset_index().rename(columns={0: 'dtype'})['dtype']
    t['unique_values_cnt_withnull'] = df.nunique(axis=0, dropna=False).reset_index().rename(columns={0: 'unique_values_cnt_withnull'})['unique_values_cnt_withnull'] # unique values including NaN

    t['unique_values_percentage'] = (t['unique_values_cnt']/df.shape[0]*100).round(2)
    t['missing_cnt'] = df.isnull().sum().reset_index().rename(columns={0: 'missing_cnt'})['missing_cnt']
    t['missing_percentage'] = ((t['missing_cnt']/ df.shape[0]).round(4)*100)
    t['empty_str_cnt'] = (df == '').sum().reset_index().rename(columns={0: 'empty_string_count'})['empty_string_count']
    t['empty_str_percentage'] = ((t['empty_str_cnt']/ df.shape[0]).round(4)*100)


    # t = t.sort_values(by='missing_percentage', ascending=False)

    # t = t.sort_values(by='unique_values_cnt', ascending=False)
    t
    print(t.shape)

    i = 0
    for feature in df.columns:
        m = df[feature].mode(dropna=True)
        median_val = df[feature].median() if df[feature].dtype in ['int64', 'float64'] else np.nan
        # mean_val = df[feature].mean() if df[feature].dtype in ['int64', 'float64'] else np.nan
        # std_val = df[feature].std() if df[feature].dtype in ['int64', 'float64'] else np.nan
        # min_val = df[feature].min() if df[feature].dtype in ['int64', 'float64'] else np.nan
        # max_val = df[feature].max() if df[feature].dtype in ['int64', 'float64'] else np.nan
        #  
        mode_val = m.values[0] if len(m) > 0 else np.nan
        mode_cnt = len(m)
        mode_values = ', '.join(map(str, m.values)) if (len(m) > 0 and len(m)<10) or (len(m) > 0) else np.nan
        unique_values = ', '.join(map(str, df[feature].unique()[0:9])) #if len(df[feature].unique())<10 else np.nan
        # if feature == 'E10':
        #     print(feature, df[feature].unique(), unique_values)
        if i == 0:
            i += 1
            tmp = pd.DataFrame([[feature, median_val, mode_val, mode_cnt, mode_values, unique_values]], columns=['feature', 'median_val', 'mode_val', 'mode_cnt', 'mode_values', 'unique_values'])
            #t = pd.concat([t, tmp], axis=0, join="inner")
            continue
        tmp = pd.concat([tmp, pd.DataFrame([[feature, median_val, mode_val, mode_cnt, mode_values, unique_values]], columns=['feature', 'median_val', 'mode_val', 'mode_cnt', 'mode_values', 'unique_values'])])

        # t['missing_percentage'] = ((df[feature].isnull().sum()/ df.shape[0]).round(4)*100)

    t = t.merge(tmp, on='feature', how='left', suffixes=('', '_y'))
    try:
        t = t.merge(df.describe(include=['int64', 'float64']).transpose().reset_index().rename(columns={'index': 'feature'}), on='feature', how='left', suffixes=('', '_y'))
    except:
        print("no numeric columns")
        pass
    is_categorical = False
    try:
        t = t.merge(df.describe(exclude=['int64', 'float64']).transpose().reset_index().rename(columns={'index': 'feature'}), on='feature', how='left', suffixes=('', '_y'))
        is_categorical = True
    except:
        print("no categorical columns")
        pass    
    t = t.drop(columns=[col for col in t.columns if col.endswith('_y')])
    # t[t['feature'] == 'E1']
    #t = t.sort_values(by='missing_percentage', ascending=False)
    if is_categorical:
        features = ['feature', 'data_type', 'count', 'unique_values_cnt', 'unique_values_percentage', 'unique_values_cnt_withnull', 'missing_cnt', 'missing_percentage',
 'empty_str_cnt', 'empty_str_percentage', 'median_val', 'mode_val', 'mode_cnt', 'mode_values', 'unique_values', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'unique', 'top', 'freq']
    else:
        features = ['feature', 'data_type', 'count', 'unique_values_cnt', 'unique_values_percentage', 'unique_values_cnt_withnull', 'missing_cnt', 'missing_percentage',
 'empty_str_cnt', 'empty_str_percentage', 'median_val', 'mode_val', 'mode_cnt', 'mode_values', 'unique_values', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return t[features]









# nunique(axis=0, dropna=True)
# The axis to use. 0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise.
# Return the number of unique values for each column, excluding NA/null values.
# This function is useful for quickly assessing the diversity of values in a DataFrame.

"""
dataframe.mode()
axis {0 or ‘index’, 1 or ‘columns’}, default 0
0 or ‘index’ : get mode of each column
1 or ‘columns’ : get mode of each row.

numeric_only
bool, default False
If True, only apply to numeric columns.

dropna
bool, default True
Don’t consider counts of NaN/NaT.
"""

def get_summary_df(df, categorical_features):
    rows, cols = df.shape
    categorical_features_df = df[categorical_features]
    categorical_features_summary_df = pd.DataFrame(columns = ['feature', 'mode', 'median', 'nunique_cnt_withnull', 'nunique_cnt_wo_null', 'unique_cnt_percentage', 'missing_cnt', 'missing_percentage', 'unique_values'])  # 'unique_value_count_withnull'  , 'val_count'
    for feature in categorical_features:
        temp = pd.DataFrame()
        temp['feature'] = [feature]
        # temp['unique_value_count_withnull'] = [len(categorical_features_df[feature].unique())]
        temp['mode'] = [categorical_features_df[feature].mode().tolist()]
        try:
            temp['median'] = [categorical_features_df[feature].median().tolist()]
        except:
            pass
        temp['nunique_cnt_withnull'] = [categorical_features_df[feature].nunique(dropna=False)]
        temp['nunique_cnt_wo_null'] = [categorical_features_df[feature].nunique(dropna=True)]
        temp['unique_cnt_percentage'] = [int(round(categorical_features_df[feature].nunique(dropna=False)/ rows, 0)*100)]
        temp['missing_cnt'] = [categorical_features_df[feature].isnull().sum()]
        temp['missing_percentage'] = [((categorical_features_df[feature].isnull().sum()/ rows).round(4)*100)]
        temp['unique_values'] = [categorical_features_df[feature].unique()]
        # temp['val_count'] = [categorical_features_df[feature].value_counts().reset_index().to_numpy()]
        categorical_features_summary_df = pd.concat([categorical_features_summary_df, temp])
    return categorical_features_summary_df.reset_index().drop('index', axis=1)

def get_stats_df(df):
    stats_df = pd.DataFrame(df.dtypes, columns=['dtypes'])
    desc_results = df.describe(include='all').transpose()
    try:
        # ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        stats_df['non_null_count'] = desc_results['count'].astype(int)
    except:
        pass
    try:
        stats_df['missing_count'] = df.isna().sum().values
    except:
        pass
    try:
        stats_df['missing_percentage'] = (df.isna().sum().values/len(df)*100).round(2)
    except:
        pass
    try:
        stats_df['uniques'] = df.nunique()
        stats_df['total_count'] = len(df)
    except:
        pass
    try:
        stats_df['unique'] = desc_results['unique']
    except:
        pass
    try:
        stats_df['top'] = desc_results['top']
    except:
        pass
    try:
        stats_df['freq'] = desc_results['freq']
    except:
        pass
    try:
        stats_df['mean'] = desc_results['mean']
        stats_df['std'] = desc_results['std']
        stats_df['min'] = desc_results['min']
        stats_df['25%'] = desc_results['25%']
        stats_df['50%'] = desc_results['50%']
        stats_df['75%'] = desc_results['75%']
        stats_df['min'] = desc_results['min']
        stats_df['max'] = desc_results['max']
    except:
       pass
    # summary_df.info()
    return  stats_df.reset_index().rename(columns={'index': 'feature'})
