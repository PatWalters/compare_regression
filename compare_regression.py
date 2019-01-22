#!/usr/bin/env python

import sys

import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import PandasTools
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from glob import glob
import pandas as pd
from bootstrap import bootstrap_error_estimate
import os

from pandas_desc import PandasDescriptors
from tqdm import tqdm


def get_r2(truth, pred):
    """
    Simple convenience function to return r**2 as a tuple to be consistent with scipy
    :param truth: true values
    :param pred: predicted values
    :return: tuple with R**2 and 0.0
    """
    return tuple([r2_score(truth, pred), 0])


def run_model(train_X, test_X, train_Y, test_Y):
    """
    Given training and test data build a model with XGBoost
    :param train_X: training x values (descriptors)
    :param test_X: test x values (descriptors)
    :param train_Y: training y values (activity values)
    :param test_Y: test y values (activity values)
    :return: list of lists with [lower bound, value, upper bound] for Spearman rho, R**2, Kendall tau
    """
    estimator = xgb.XGBRegressor()
    estimator.fit(train_X, train_Y)
    pred_Y = estimator.predict(test_X)
    res = []
    for stat_func, stat_name in [(spearmanr, "Spearman"), (get_r2, "R**2"), (kendalltau, "Kendal")]:
        stat_val = stat_func(test_Y, pred_Y)[0]
        bs_min, bs_max = bootstrap_error_estimate(test_Y, pred_Y, stat_func)
        res.append([stat_name, bs_min, stat_val, bs_max])
    return res


def split_train_test(input_df, train_idx, test_idx, offset, activity_column):
    """
    Given an input dataframe and a set of indices defining training and test sets, return x and y for model building
    :param input_df: input dataframe
    :param train_idx: indices of the training and
    :param test_idx:
    :param offset:
    :param activity_column:
    :return:
    """
    train_df = input_df.iloc[train_idx]
    test_df = input_df.iloc[test_idx]
    train_x = train_df.values[0::, offset::]
    test_x = test_df.values[0::, offset::]
    train_y = train_df[activity_column].values
    test_y = test_df[activity_column].values
    return train_x, test_x, train_y, test_y


def compare_models(input_file_name):
    print(input_file_name)
    target, _ = os.path.splitext(input_file_name)

    # read the SD file and generate morgan fingerprints and RDKit 2D descriptors
    df = PandasTools.LoadSDF(input_file_name)
    df.pIC50 = [float(x) for x in df.pIC50]
    df['SMILES'] = [Chem.MolToSmiles(x) for x in df.ROMol]
    pandas_descriptors = PandasDescriptors(['morgan2', 'descriptors'])
    desc_df = pandas_descriptors.from_dataframe(df, smiles_column='SMILES', name_column='ChEMBL_ID')
    rows, _ = desc_df.shape

    # read the correspond CDDD descriptors from disk
    cddd_df = pd.read_csv(input_file_name.replace(".sdf", "_cddd.csv"))
    cddd_df.insert(1, "pIC50", df.pIC50.values)

    # run 10 folds of cross validation for each of the two descriptor sets, save the value, lower bound,
    # and upper bound for Spearman rho, Pearson r**2, and Kendall tau
    cv_cycles = 10
    ss = ShuffleSplit(n_splits=cv_cycles, test_size=0.3, random_state=0)
    res = []
    for train_idx, test_idx in tqdm(ss.split(range(0, rows)), total=cv_cycles):
        train_x, test_x, train_y, test_y = split_train_test(desc_df, train_idx, test_idx, 6, "pIC50")
        desc_res = run_model(train_x, test_x, train_y, test_y)
        for row in desc_res:
            res.append([target, "DESC"] + row)

        train_x, test_x, train_y, test_y = split_train_test(cddd_df, train_idx, test_idx, 4, "pIC50")
        cddd_res = run_model(train_x, test_x, train_y, test_y)
        for row in cddd_res:
            res.append([target, "CDDD"] + row)
    return res


def main():
    out_list = []
    for filename in glob("*.sdf"):
        out_list += compare_models(filename)
    out_df = pd.DataFrame(out_list, columns=["Target", "Method", "Stat_Type", "Stat_LB", "Stat_Val", "Stat_UB"])
    out_df.to_csv("comparison.csv", index=False)


if __name__ == "__main__":
    main()
