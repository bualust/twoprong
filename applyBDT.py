from alive_progress import alive_bar
from xgboost import XGBClassifier
import awkward as ak
import ROOT as root
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
    file = root.TFile.Open("tree_tester_"+file_name, "RECREATE")
    tree = root.TTree("nominal","nominal")
    bdt_score = []
    tree.Branch('bdt_score', 'bdt_score', 'bdt_score/D')
    with alive_bar(len(input_file)) as bar:
        for count,ev in enumerate(input_file):
            bar()
            lj_score = []
            data = {}
            for t21,t32,s12,s23,s34,ecf1 in zip(ev["ljet_tau21"],
                                                ev["ljet_tau32"],
                                                ev["ljet_Split12"],
                                                ev["ljet_Split23"],
                                                ev["ljet_Split34"],
                                                ev["ljet_ECF1"]):
                data["ljet_tau21_0"] = t21
                data["ljet_tau32_0"] = t32
                data["ljet_Split12_0"] = s12
                data["ljet_Split23_0"] = s23
                data["ljet_Split34_0"] = s34
                data["ljet_ECF1_0"] = ecf1
                data = ak.to_pandas(data)
                if data.shape[1]!=6: continue
                test_proba = bst.predict_proba(data)
                lj_score.append(test_proba[:,1])
            bdt_score.append(lj_score)
            tree.Fill()
    tree.Write()

    return 0

#main
if __name__ == '__main__':
    main()
