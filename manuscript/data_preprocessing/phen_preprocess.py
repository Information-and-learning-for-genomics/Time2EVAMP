import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-train_ind", "--train-ind", help="Training sample indices corresponding to eids", default=None)
parser.add_argument("-test_ind", "--test-ind", help="Test sample indices corresponding to eids", default=None)
parser.add_argument(
    "-participant_table",
    "--participant-table",
    help="Comma-separated table containing the date of the outcome of interest, birth year, and assessment date. The table must contain the columns 'eid', 'y', 'birth', and 'assessment'.",
    default=None
)
parser.add_argument("-out", "--out", help="Output prefix")
args = parser.parse_args()

train_indlist_fpath = args.train_ind
test_indlist_fpath = args.test_ind
participant_table_fpath = args.participant_table
out = args.out

print("----- phen_preprocess.py -----")
print("--out", out)
print("--train-ind", train_indlist_fpath)
print("--test-ind", test_indlist_fpath)
print("--participant-table", participant_table_fpath)

df_train_ind = pd.read_table(train_indlist_fpath, header=None, names=["IID"])
df_test_ind = pd.read_table(test_indlist_fpath, header=None, names=["IID"])
df = pd.read_table(participant_table_fpath, sep=",") # Table has to contain columns 'eid,y,birth,assesment'

print((df["y"] == "1900-00-00").sum())
print((df["y"] == "1901-01-01").sum())
print((df["y"] == "1902-02-02").sum())
print((df["y"] == "1903-03-03").sum())
print((df["y"] == "1909-09-09").sum())
print((df["y"] == "2037-07-07").sum())
df[df["y"] == "1900-00-00"] = None
df[df["y"] == "1901-01-01"] = None
df[df["y"] == "1902-02-02"] = None
df[df["y"] == "1903-03-03"] = None
df[df["y"] == "1909-09-09"] = None
df[df["y"] == "2037-07-07"] = None

df["y"] = pd.to_datetime(df["y"], errors='coerce').dt.year
df["assessment_year"] = pd.to_datetime(df["assessment"], errors='coerce').dt.year

# uncensored individuals
df["y"] = (df["y"] - df["birth"])
df["event"] = np.ones(len(df)).astype(int)

# censored individuals
mask = df["y"].isnull()
df.loc[mask, "y"] = (df.loc[mask,"assessment_year"] - df.loc[mask,"birth"])
df.loc[mask,"event"] = 0

df["age_at_assesment"] = (df["assessment_year"] - df["birth"])
df = df[["eid", "y", "event", "age_at_assesment"]]
df = df.rename(columns={"eid":"IID"})
df["FID"] = df["IID"]

df_train = pd.merge(df_train_ind, df, on="IID", how="inner")
df_test = pd.merge(df_test_ind, df, on="IID", how="inner")
print("Number of cases in train:", df_train["event"].sum())
print("Number of cases in test:", df_test["event"].sum())

print("\n...saving train/test phenotype files")
df_train.columns = df_train.columns.str.upper()
df_test.columns = df_test.columns.str.upper()
df_train[["IID", "FID", "Y", "EVENT", "AGE_AT_ASSESMENT"]].to_csv(out+"_train.phen", index=False, sep="\t")
df_test[["IID", "FID", "Y", "EVENT", "AGE_AT_ASSESMENT"]].to_csv(out+"_test.phen", index=False, sep="\t")