from alive_progress import alive_bar
from xgboost import XGBClassifier
import awkward as ak
import ROOT as root
import numpy as np
import uproot
import time

def main():

    bst = XGBClassifier()
    bst.load_model("twoprong.json")

    variables = ["ljet_tau21","ljet_tau32","ljet_Split12","ljet_Split23",
                 "ljet_Split34","ljet_ECF1"]

    apply_BDT_score(bst, variables)

def apply_BDT_score(bst, variables):


    file_name = "800031.root"
    #file_name = "364703.root"
    input_file = uproot.concatenate("../"+file_name+":nominal",variables,
                                    allow_missing=True)

    tau_21 = ak.Array(input_file["ljet_tau21"])

    input_file = ak.to_pandas(input_file[variables])
    input_file = rename_variables(input_file,variables)
    test_proba = bst.predict_proba(input_file)
    bdt_scores = ak.Array(test_proba[:,0])
    counts = ak.num(tau_21)
    unflattened_bdt_score = ak.unflatten(bdt_scores,counts)

    with uproot.recreate("tree_tester_"+file_name) as file:
        file["nominal"] = {"bdt_score":unflattened_bdt_score}

    return 0

#remap data to have same variables names as in training
def rename_variables(input_file, bdt_vars):
    for var in bdt_vars: input_file[var+"_0"] = input_file[var]
    input_file = input_file.drop(columns=bdt_vars)
    return input_file

#main
if __name__ == '__main__':
    main()
