#!/usr/bin/env python

import sys
import pandas as pd
from glob import glob


def reformat_cddd(infile_name):
    df = pd.read_csv(infile_name)
    print(df.shape)
    name_list = [x[1] for x in df['smiles'].str.split()]
    df.insert(0,"Name",name_list)
    good_cols = ["Name"] + [x for x in df.columns if x.startswith("cddd")]
    good_df = df[good_cols]
    good_df.to_csv(infile_name,index=False)


for file_name in sorted(glob("*cddd.csv")):
    print(file_name)
    reformat_cddd(file_name)