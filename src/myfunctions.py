import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import dump, load
import json
import numpy as np
from random import randint
from sklearn import metrics

def save_fraction_csv(csv_path: str, fraction_folder: str, fraction_fiename: str, fraction_size: float):
    # read the csv file
    df = pd.read_csv(csv_path)

    # calculate the split index
    split_idx = int(len(df) * fraction_size)

    # split the dataframe
    df_fraction = df[:split_idx]
    df_remainder = df[split_idx:]

    # save the fraction dataframe
    df_fraction.to_csv(f'{fraction_folder}/{fraction_fiename}', index=False)

    # save the remainder dataframe
    df_remainder.to_csv(csv_path, index=False)


def print_unique(ds):
    print(ds.nunique())


def drop_column(df, atr_name: str) -> None:
    df.drop(atr_name, axis=1, inplace=True)


def impute_na(ds, variable: str, value):
    return ds[variable].fillna(value)


def print_nans(ds):
    for x in ds.columns:
        if ds[x].isnull().sum() != 0:
            print(x, ds[x].isnull().sum())


def print_missing_pers(ds, atr_name: str):
    mean = ds[atr_name].isnull().mean()
    print(f"Missing values in {atr_name} : {mean * 100} %")


def get_mean_impute_dict(ds, mean_impute_columns):
    mean_impute_values = dict()
    for column in mean_impute_columns:
        mean_impute_values[column] = ds[column].mean()
    return mean_impute_values


def get_mode_impute_dict(ds, mode_impute_columns: list):
    mode_impute_values = dict()
    for column in mode_impute_columns:
        mode_impute_values[column] = ds[column].mode()
    return mode_impute_values


def impute_by_dict(ds, impute_values: dict, is_category: False) -> None:
    for column in impute_values.keys():
        impute_val = impute_values[column][0]
        if is_category:
            impute_val = impute_val[0]
        ds[column] = impute_na(ds, column, impute_val)


def impute_by_mode(ds, columns: list, is_category: False) -> dict:
    impute_values = get_mode_impute_dict(ds, columns)
    impute_by_dict(ds, impute_values, is_category)
    return impute_values


def impute_by_mean(ds, columns: list) -> None:
    impute_values = get_mean_impute_dict(ds, columns)
    impute_by_dict(ds, impute_values)


def impute_by_arbitrary(ds, columns: list, arb_val) -> dict:
    # Create a dictionary with columns as keys and arb_val as values
    arb_dict = {column: arb_val for column in columns}

    for column in columns:
        ds[column] = impute_na(ds, column, arb_val)

    return arb_dict


def random_sample_imputation(ds, atr_name: str, atr_imputed="", random_state=0):
    if atr_imputed == "":
        atr_imputed = atr_name
    ds[atr_imputed] = ds[atr_name].copy()
    random_sample_train = ds[atr_name].dropna().sample(
        ds[atr_name].isnull().sum(), random_state=random_state)
    random_sample_train.index = ds[ds[atr_name].isnull()].index
    ds.loc[ds[atr_imputed].isnull(), atr_imputed] = random_sample_train


def impute_by_random(ds, columns: list, random_state=0) -> list:
    for column in columns:
        random_sample_imputation(ds, column, random_state=random_state)

    return columns


def diagnostic_plots(df, variable):
    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()


def save_model(model, path: str, filename: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    # save the model to disk
    dump(model, f'{path}/{filename}')


def save_dict(dict_to_save: dict, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, pd.Series) else v for k, v in dict_to_save.items()}, f)


def load_dict(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def save_modes_dict(modes_dict: dict) -> None:
    save_dict(modes_dict, "./feature_engineering_data/modes.json")


def load_modes_dict() -> dict:
    return load_dict("./feature_engineering_data/modes.json")


def save_arbs_dict(arb_dict: dict) -> None:
    save_dict(arb_dict, "./feature_engineering_data/arbitrary.json")


def load_arbs_dict() -> dict:
    return load_dict("./feature_engineering_data/arbitrary.json")


def load_list(path: str) -> list:
    with open(path, 'r') as f:
        return json.load(f)


def save_list(list_to_save: list, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(list_to_save, f)


def save_rand_col_list(rand_cols: list) -> None:
    save_list(rand_cols, "./feature_engineering_data/random_imp.json")


def load_rand_col_list() -> list:
    return load_list("./feature_engineering_data/random_imp.json")


def save_del_cols(del_cols: list) -> None:
    save_list(del_cols, "./feature_engineering_data/delete_cols.json")


def load_del_cols() -> list:
    return load_list("./feature_engineering_data/delete_cols.json")


def save_outliers(outl_dict: dict) -> None:
    save_dict(outl_dict, "./feature_engineering_data/outlier_params.json")


def load_outliers() -> dict:
    return load_dict("./feature_engineering_data/outlier_params.json")


def save_categorical_encoding(categ_dict: dict) -> None:
    save_dict(categ_dict, "./feature_engineering_data/categorical_encoding.json")


def load_categorical_encoding() -> dict:
    return load_dict("./feature_engineering_data/categorical_encoding.json")


def save_scaler(scaler) -> None:
    # Save the fitted scaler
    dump(scaler, './feature_engineering_data/scaler.joblib')


def load_scaler():
    # Load the fitted scaler
    return load('./feature_engineering_data/scaler.joblib')


def load_model(path: str):
    # load the model from disk
    loaded_model = load(path)
    return loaded_model


def show_high_correlations(df, threshold):
    correlations = df.corr()
    cols = correlations.columns
    print(f"Correlations ( > {threshold} ) inside dataset:  ")
    for i in range(0, len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(correlations.iloc[i, j]) > threshold:
                print(f'{cols[i]} -> {cols[j]}: {correlations.iloc[i, j]}')

def plot_attribute_density(df, atr_name:str) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,5))
    sns.distplot(df[atr_name], kde=True, bins=20)
    plt.axvline(df[atr_name].mean(), c='red', linestyle='dashed', label='Mean')
    plt.text(0.49, 0.8, f'Average Price = {df[atr_name].mean():.2f}', transform=plt.gca().transAxes, color='red', fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white',alpha=0.5))
    plt.legend()
    plt.show()

def plot_attrs_frequency(df, selected_cols: list) -> None:
    colors = ['mediumvioletred']
    for i in range(len(selected_cols)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    num_subplots = len(selected_cols)
    num_rows = (num_subplots - 1) // 5 + 1
    num_cols = min(5, num_subplots)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4), facecolor='white')
    fig.suptitle("Histograms of Various Features", size=24)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_subplots:
                sns.histplot(df[selected_cols[idx]], ax=axes[i, j], color=colors[idx], kde=True, bins=8)
                axes[i, j].set_xlabel(selected_cols[idx], fontsize=12)  # Set x-axis label font size
                axes[i, j].set_ylabel("Frequency", fontsize=12)  # Set y-axis label font size

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_attr_distr_by_catcols(df, atr_name: str, selected_cols: list) -> None:
    num_subplots = len(selected_cols)
    num_rows = (num_subplots - 1) // 3 + 1
    num_cols = min(3, num_subplots)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), facecolor='white')
    fig.suptitle(f"Avg {atr_name} for Various Categorical Features", size=20)
    colors = sns.light_palette('mediumvioletred', n_colors=len(selected_cols) + 1, reverse=True)
    # If the axes object is not a 2D array, make it a 2D array for consistency
    if num_rows == 1 or num_cols == 1:
        axes = np.array(axes).reshape(num_rows, num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_subplots:
                sns.boxplot(x=selected_cols[idx], y=atr_name, data=df, ax=axes[i][j], palette=[colors[idx]])
                axes[i][j].set_title(f'{atr_name} Distribution by {selected_cols[idx]}', fontsize=14)
                axes[i][j].set_xlabel(selected_cols[idx], fontsize=12)
                axes[i][j].set_ylabel(atr_name, fontsize=12)
                axes[i][j].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_attr_polym_line_vs_num_feature(df, attr_name: str, selected_cols: list) -> None:
    num_subplots = len(selected_cols)
    num_rows = (num_subplots - 1) // 3 + 1
    num_cols = min(3, num_subplots)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), facecolor='white')
    fig.suptitle(f"Scatter Plots of Numerical Features vs {attr_name} with Polynomial Lines", size=20)
    palette = sns.husl_palette(n_colors=len(selected_cols), s=0.7, l=0.6)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_subplots:
                sns.scatterplot(x=selected_cols[idx], y=attr_name, data=df, ax=axes[i, j], color=palette[idx])
                sns.regplot(x=selected_cols[idx], y=attr_name, data=df, ax=axes[i, j], scatter=False, order=2,
                            color=palette[idx], ci=None)
                axes[i, j].set_xlabel(selected_cols[idx], fontsize=12)
                axes[i, j].set_ylabel(attr_name, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def test_regression_models(X_train, y_train, X_test, y_test, models:list, metric):
    best_model = None
    best_score = float('inf') # start with infinity so that any score will be better
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        if score < best_score:
            best_score = score
            best_model = model
        print("\n" + str(model) + ":\n"+str(metric)+":\n" + str(score))
    return best_model


def plot_feature_importances(model, X_columns):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    # Only plot the feature importances of the top n features
    n_features = len(X_columns)
    plt.bar(range(n_features), importances[indices], color="b", align="center")
    plt.xticks(range(n_features), np.array(X_columns)[indices], rotation='vertical')
    plt.xlim([-1, n_features])
    plt.show()


def plot_permutation_importances(X_columns, importances, X_train, indices, std):
    plt.figure(figsize=(20, 7))
    plt.title("Feature permutation importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [X_columns[indices[f]] for f in range(X_train.shape[1])])
    plt.xticks(rotation=70)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()