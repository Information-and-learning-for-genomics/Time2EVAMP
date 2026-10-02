from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import argparse

def normalize_matrix(X):
    n, m = X.shape
    for i in range(m):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X
    
def adjust_covariant(X_path, iids_path, cov_fpath, model_type, saving_path):
    X = np.load(X_path)
    df_iids = pd.DataFrame({"eid": np.loadtxt(iids_path, dtype=int).tolist()})
    df_sex_age =pd.read_table(cov_fpath, sep=",") 
    df_sex_age = pd.merge(df_iids, df_sex_age, how="left", on="eid")
    df_sex_age["age_exp"] = np.exp(df_sex_age["assessment_age"])
    sex_age = df_sex_age[["assessment_age", "sex"]].to_numpy()
    sex_age_exp = df_sex_age[["assessment_age", "sex", "age_exp"]].to_numpy()
    n, m = X.shape
    print(n,m, flush=True)

    for i in range(m):
        yi = X[:, i]
        if model_type == "linear":
            model = LinearRegression().fit(sex_age, yi)
            y_pred = model.predict(sex_age)
        elif model_type == "exponential":
            model = LinearRegression().fit(sex_age_exp, yi)
            y_pred = model.predict(sex_age_exp)
        else:
            raise ValueError("Model type not supported!")
        
        X[:, i] = yi - y_pred
    
    X = normalize_matrix(X)
    
    print(f"Saving result in {saving_path}", flush=True)
    np.savez(f"{saving_path}.npy", X)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-X", "--X", help="Design matrix", default=None)
    parser.add_argument("-indlist", "--indlist", help="List of IDs matching the order in X", default=None)
    parser.add_argument("-model", "--model", help="Model (linear or exponential)", default="linear")
    parser.add_argument("-cov", "--cov", help="Covariate table. Must contain the columns: eid, assessment_age, and sex", default=None)
    parser.add_argument("-out", "--out", help="Output prefix")
    args = parser.parse_args()
    
    adjust_covariant(args.X, args.indlist, args.cov, args.model, args.out)
    