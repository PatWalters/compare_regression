#!/usr/bin/env python

import sys
import pandas as pd
from rdkit.Chem import PandasTools
import numpy as np
import os
import glob
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import xgboost as xgb


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


def read_activity_data(sdf_name, name_tag, activity_tag):
    """
    Read activity data from the an SD file
    :param sdf_name: input sd file
    :param name_tag: the SD tag with the molecule name
    :param activity_tag: the SD tag with activity data (will be converted to float)
    :return: dataframe with Name and activity
    """
    sdf_df = PandasTools.LoadSDF(sdf_name)
    name_list = sdf_df[name_tag]
    activity_list = [float(x) for x in sdf_df[activity_tag]]
    return pd.DataFrame(np.transpose([name_list, activity_list]), columns=["Name", activity_tag])


def read_descriptors(descriptor_file_name, name_column, start_idx):
    """
    Read molecular descriptors from a csv file
    :param descriptor_file_name: input descriptor file name
    :param name_column: column defining the molecule name
    :param start_idx: column defining the first descriptor, all subsequent columns are assumed to be descriptors
    :return:
    """
    descriptor_df = pd.read_csv(descriptor_file_name)
    output_columns = [name_column] + list(descriptor_df.columns[start_idx:])
    output_df = descriptor_df[output_columns]
    output_columns[0] = "Name"
    output_df.columns = output_columns
    return output_df


def read_data(sdf_name, suffix):
    """
    Read a activity data and descriptors, descriptor file is assumed to be sdf base name + "_" + suffix + ".csv"
    :param sdf_name: name of the sd file
    :param suffix: suffix for the particular descriptor type
    :return:
    """
    base_name, _ = os.path.splitext(sdf_name)
    csv_name = f"{base_name}_{suffix}.csv"
    desc_df = read_descriptors(csv_name, 'Name', 1)
    act_df = read_activity_data(sdf_name, name_tag="ChEMBL_ID", activity_tag="pIC50")
    act_df = act_df.merge(desc_df, on="Name")
    return act_df


def cv_models(df, base_name, splits, suffix):
    out_list = []
    for cycle_num,[train_idx, test_idx] in enumerate(tqdm(splits, desc=suffix)):
        test_names = df.Name.iloc[test_idx]
        train_x, test_x, train_y, test_y = split_train_test(df, train_idx, test_idx, 1, "pIC50")
        estimator = xgb.XGBRegressor()
        estimator.fit(train_x, train_y)
        pred_y = estimator.predict(test_x)
        for name, expt, pred in zip(test_names, test_y, pred_y):
            out_list.append([suffix, base_name, cycle_num, name, expt, pred])
    return out_list


def main():
    output_list = []
    for file_name in sorted(glob.glob("A*.sdf")):
        print(file_name)
        base_name, _ = os.path.splitext(file_name)
        df_desc = read_data(file_name, "desc")
        df_cddd = read_data(file_name, "cddd")
        rows, _ = df_desc.shape
        cv_cycles = 10
        ss = ShuffleSplit(n_splits=cv_cycles, test_size=0.3, random_state=0)
        splits = list(ss.split(range(0, rows)))
        output_list += cv_models(df_desc, base_name, splits, "desc")
        output_list += cv_models(df_cddd, base_name, splits, "cddd")
    output_df = pd.DataFrame(output_list, columns=["Method", "Dataset", "CV_Cycle", "Name", "Truth", "Pred"])
    output_df.to_csv("xgboost_result.csv", index=False)



main()