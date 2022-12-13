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
                 "ljet_flavourlabel","ljet_Split34","ljet_ECF1"]

    apply_BDT_score(bst, variables)

def apply_BDT_score(bst, variables):


    file_name = "800031.root"
    #file_name = "364703.root"
    input_file = uproot.concatenate("../"+file_name+":nominal",variables,
                                    allow_missing=True)
    bdt_score = []
    with alive_bar(len(input_file)) as bar:
        for count,ev in enumerate(input_file):
            if count>1000: break
            bar()
            array_lenght = len(ev["ljet_tau21"])
            lj_score = np.zeros(array_lenght,dtype=np.float64)
            lj_index = 0
            for t21,t32,s12,s23,s34,ecf1 in zip(ev["ljet_tau21"],
                                                ev["ljet_tau32"],
                                                ev["ljet_Split12"],
                                                ev["ljet_Split23"],
                                                ev["ljet_Split34"],
                                                ev["ljet_ECF1"]):
                data = {}
                data["ljet_tau21_0"] = t21
                data["ljet_tau32_0"] = t32
                data["ljet_Split12_0"] = s12
                data["ljet_Split23_0"] = s23
                data["ljet_Split34_0"] = s34
                data["ljet_ECF1_0"] = ecf1
                lj_index+=1
                data = ak.to_pandas(data)
                if data.shape[1]!=6: continue
                test_proba = bst.predict_proba(data)
                jet_score = test_proba[:,0]
                lj_score[lj_index-1] = jet_score
            bdt_score.append(lj_score)
    with uproot.recreate("tree_tester_"+file_name) as file:
        file["nominal"] = {"bdt_score": ak.Array(bdt_score)}

    return 0

#main
if __name__ == '__main__':
    main()
