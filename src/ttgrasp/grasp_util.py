import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


def get_df_grasp(df):

    t = df.nunique(axis=0, dropna=True).reset_index()
    t.columns = ['feature', 'unique_values_cnt']
    t['data_type'] = df.dtypes.reset_index().rename(columns={0: 'dtype'})['dtype']
    t['unique_values_cnt_withnull'] = (df.nunique(axis=0, dropna=False).reset_index().rename(columns={0: 'unique_values_cnt_withnull'})['unique_values_cnt_withnull']) # unique values including NaN
    t['unique_values_percentage'] = (t['unique_values_cnt'] / df.shape[0] * 100).round(2)
    t['missing_cnt'] = (df.isnull().sum().reset_index().rename(columns={0: 'missing_cnt'})['missing_cnt'])
    t['missing_percentage'] = ((t['missing_cnt'] / df.shape[0]).round(4) * 100)
    t['empty_str_cnt'] = ((df == '').sum().reset_index().rename(columns={0: 'empty_string_count'})['empty_string_count'])
    t['empty_str_percentage'] = ((t['empty_str_cnt'] / df.shape[0]).round(4) * 100)

    print(t.shape)

    rows = []
    for feature in df.columns:
        m = df[feature].mode(dropna=True)
        median_val = df[feature].median() if df[feature].dtype in ['int64', 'float64'] else np.nan
        mode_val = m.values[0] if len(m) > 0 else np.nan
        mode_cnt = len(m)
        mode_values = ', '.join(map(str, m.values)) if len(m) > 0 else np.nan
        unique_values = ', '.join(map(str, df[feature].unique()[:9]))
        n_unique = df[feature].nunique(dropna=False)
        if n_unique <= 5:
            vc = df[feature].value_counts(dropna=False).sort_index()
            value_counts_dist = ' | '.join([f"{v}:{c}({c / len(df) * 100:.1f}%)" for v, c in vc.items()])
        else:
            value_counts_dist = f"high cardinality, that is ({n_unique} unique values.)"

        rows.append([feature, median_val, mode_val, mode_cnt, mode_values, unique_values, value_counts_dist])

    tmp = pd.DataFrame(rows, columns=['feature', 'median_val', 'mode_val', 'mode_cnt', 'mode_values', 'unique_values', 'value_counts_dist'])

    t = t.merge(tmp, on='feature', how='left', suffixes=('', '_y'))

    try:
        t = t.merge(
            df.describe(include=['int64', 'float64'])
            .transpose()
            .reset_index()
            .rename(columns={'index': 'feature'}),
            on='feature', how='left', suffixes=('', '_y')
        )
    except Exception:
        print("no numeric columns")

    is_categorical = False
    try:
        t = t.merge(
            df.describe(exclude=['int64', 'float64'])
            .transpose()
            .reset_index()
            .rename(columns={'index': 'feature'}),
            on='feature', how='left', suffixes=('', '_y')
        )
        is_categorical = True
    except Exception:
        print("no categorical columns")

    t = t.drop(columns=[col for col in t.columns if col.endswith('_y')])

    if is_categorical:
        features = [
            'feature', 'data_type', 'count', 'unique_values_cnt', 'unique_values_percentage',
            'unique_values_cnt_withnull', 'missing_cnt', 'missing_percentage',
            'empty_str_cnt', 'empty_str_percentage', 'median_val', 'mode_val', 'mode_cnt',
            'mode_values', 'unique_values', 'value_counts_dist', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
            'unique', 'top', 'freq'
        ]
    else:
        features = [
            'feature', 'data_type', 'count', 'unique_values_cnt', 'unique_values_percentage',
            'unique_values_cnt_withnull', 'missing_cnt', 'missing_percentage',
            'empty_str_cnt', 'empty_str_percentage', 'median_val', 'mode_val', 'mode_cnt',
            'mode_values', 'unique_values', 'value_counts_dist', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
        ]

    # Only include columns that actually exist (guards against mixed-type DataFrames
    # where some describe() columns may be absent)
    features = [f for f in features if f in t.columns]
    return t[features]


def get_summary_df(df, categorical_features):
    rows, cols = df.shape
    categorical_features_df = df[categorical_features]
    categorical_features_summary_df = pd.DataFrame(
        columns=[
            'feature', 'mode', 'median', 'nunique_cnt_withnull', 'nunique_cnt_wo_null',
            'unique_cnt_percentage', 'missing_cnt', 'missing_percentage', 'unique_values'
        ]
    )
    for feature in categorical_features:
        temp = pd.DataFrame()
        temp['feature'] = [feature]
        temp['mode'] = [categorical_features_df[feature].mode().tolist()]
        try:
            # FIX 3: .median() returns a scalar, not a list — removed the erroneous .tolist() call.
            temp['median'] = [categorical_features_df[feature].median()]
        except Exception:
            temp['median'] = [np.nan]
        temp['nunique_cnt_withnull'] = [categorical_features_df[feature].nunique(dropna=False)]
        temp['nunique_cnt_wo_null'] = [categorical_features_df[feature].nunique(dropna=True)]
        temp['unique_cnt_percentage'] = [
            int(round(categorical_features_df[feature].nunique(dropna=False) / rows, 0) * 100)
        ]
        temp['missing_cnt'] = [categorical_features_df[feature].isnull().sum()]
        temp['missing_percentage'] = [
            ((categorical_features_df[feature].isnull().sum() / rows).round(4) * 100)
        ]
        temp['unique_values'] = [categorical_features_df[feature].unique()]
        categorical_features_summary_df = pd.concat([categorical_features_summary_df, temp])

    return categorical_features_summary_df.reset_index().drop('index', axis=1)


def get_stats_df(df):
    stats_df = pd.DataFrame(df.dtypes, columns=['dtypes'])
    desc_results = df.describe(include='all').transpose()

    try:
        # FIX 4: desc_results['count'] can contain NaN for object columns, causing
        # .astype(int) to raise. Use pd.to_numeric with errors='coerce' then fillna.
        stats_df['non_null_count'] = (
            pd.to_numeric(desc_results['count'], errors='coerce')
            .fillna(0)
            .astype(int)
        )
    except Exception:
        pass
    try:
        stats_df['missing_count'] = df.isna().sum().values
    except Exception:
        pass
    try:
        stats_df['missing_percentage'] = (df.isna().sum().values / len(df) * 100).round(2)
    except Exception:
        pass
    try:
        stats_df['uniques'] = df.nunique()
        stats_df['total_count'] = len(df)
    except Exception:
        pass
    try:
        stats_df['unique'] = desc_results['unique']
    except Exception:
        pass
    try:
        stats_df['top'] = desc_results['top']
    except Exception:
        pass
    try:
        stats_df['freq'] = desc_results['freq']
    except Exception:
        pass
    try:
        stats_df['mean'] = desc_results['mean']
        stats_df['std'] = desc_results['std']
        stats_df['min'] = desc_results['min']
        stats_df['25%'] = desc_results['25%']
        stats_df['50%'] = desc_results['50%']
        stats_df['75%'] = desc_results['75%']
        stats_df['max'] = desc_results['max']
    except Exception:
        pass

    return stats_df.reset_index().rename(columns={'index': 'feature'})