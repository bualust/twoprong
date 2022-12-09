import math
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import category_encoders as ce
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from xgboost import XGBClassifier

def main():

    LJVAR = ["ljet_tau21","ljet_tau32","ljet_Split12","ljet_Split23",
             "ljet_flavourlabel","ljet_Split34","ljet_ECF1"]
    VARS = LJVAR
    large_jet_vars = ["ljet_tau21","ljet_tau32","ljet_Split12","ljet_Split23",
                       "ljet_flavourlabel","ljet_Split34","ljet_ECF1"]
    #                   "ljetGA_trackjets_indeces"]
    #subjets_vars = ["tjet_pt","tjet_eta","tjet_phi"]

    variables = large_jet_vars #+subjets_vars

    #training data
    train_file = "../800031.root"
    trainDF = get_file(train_file, variables, large_jet_vars)

    print("large_jet_vars\n",large_jet_vars)
    print("variables\n",variables)
    #list of variables to train on
    inp_feat = input_features(variables)
    #inp_feat.append("dphi_subjets")
    #inp_feat.append("deta_subjets")

    #plot features bf training
    plot_features(trainDF,inp_feat)

    print("Training over input features\n",inp_feat)
    print(trainDF)

    #fit model
    bst = XGBClassifier()
    bst.set_params(eval_metric=['error', 'logloss','auc'],
                   max_depth=3, early_stopping_rounds=10)

    #split labeled sample in train and test
    X, X_test, Y, Y_test = train_test_split(trainDF[inp_feat],
                                            trainDF["ljet_flavourlabel_0"],
                                            test_size=0.05, random_state=10)

    eval_set = [(X,Y),(X_test, Y_test)]
    bst.fit(X,Y,eval_set=eval_set,verbose=False)
    bst.save_model('twoprong.json')

    get_feature_ranking(inp_feat,bst)
    get_accuracy(bst,X_test,Y_test)
    get_perf_plots(bst,X,Y,X_test, Y_test)
    get_class_probabilities_lab(bst, trainDF, inp_feat)
    get_class_probabilities_unlab(bst, inp_feat, VARS, LJVAR)



def get_file(file, variables,large_jet_vars):

    train_file = uproot.concatenate(file+":nominal",variables,
                                    allow_missing=True)

    #dictionary of akward array
    vect_trainDF = ak.to_pandas(train_file)

    #dataframe with flattened vectors
    trainDF = {}
    for var in large_jet_vars:
        flat_vector(vect_trainDF, 0, var, trainDF, variables, 0)
    #associate_subjets(vect_trainDF, subjets_vars, trainDF, variables)

    convert_label(trainDF)
    #set_dummy_noentry(trainDF,variables)

    trainDF = select_2P_1P(trainDF)
    trainDF = pd.DataFrame(trainDF)
    trainDF = shuffle(trainDF)
    #trainDF = trainDF.drop("ljetGA_trackjets_indeces_0",axis=1)
    #trainDF = add_features(trainDF)

    return trainDF

def convert_label(trainDF):

    new_label = []
    for ev in trainDF["ljet_flavourlabel_0"]:
        if ev==55: new_label.append(0)
        elif ev==44: new_label.append(0)
        elif ev==51: new_label.append(1)
        elif ev==41: new_label.append(1)
        else: new_label.append(-99)
    new_label = ak.Array(new_label)
    trainDF["ljet_flavourlabel_0"] = new_label

def set_dummy_noentry(trainDF, variables):

    for var in variables:
        if "tjet" not in var: continue
        new_label = []
        for ev in trainDF[var]:
            if math.isnan(ev): new_label.append(-99)
            else: new_label.append(ev)
        new_label = ak.Array(new_label)
        trainDF[var] = new_label

def select_2P_1P(trainDF):

    pandas_trainDF = ak.to_pandas(trainDF)
    trainDF_2P = pandas_trainDF.query("ljet_flavourlabel_0==0")
    trainDF_1P = pandas_trainDF.query("ljet_flavourlabel_0==1")
    print("=== Number of 2P events\n",len(trainDF_2P))
    print("=== Number of 1P events\n",len(trainDF_1P))
    DFs = [trainDF_2P,trainDF_1P]
    trainDF = pd.concat(DFs)
    return trainDF

def input_features(variables):
    input_feat = []
    for var in variables:
        if "ljetGA_trackjets_indeces" in var: continue
        if "flavourlabel" not in var: input_feat.append(var)
    return input_feat

#flatten vectors variables
def flat_vector(vect_data_frame, index, name, data_frame, variables, index_name):
    var_arr = vect_data_frame[name][:,index]
    var_arr = ak.Array(var_arr)
    new_name = name+"_"+str(index_name)
    data_frame[new_name] = var_arr
    for i in range(len(variables)):
        if variables[i]==name: variables[i]=new_name; break
    if(index_name!=0): variables.append(new_name)

def associate_subjets(vect_trainDF, subjets_vars, trainDF, variables):
    indices = [0,1]
    for idx in indices:
        for var in subjets_vars:
            var_arr = []
            for index, row in vect_trainDF.iterrows():
                if index[1]!=0: continue
                if index[2]!=idx: continue
                subjets_idx = row["ljetGA_trackjets_indeces"]
                subjets = row[var]
                var_arr.append(subjets)
            var_arr = ak.Array(var_arr)
            new_name = var+"_"+str(idx)
            trainDF[new_name] = var_arr
            for i in range(len(variables)):
                if variables[i]==var: variables[i]=new_name; break
            if(idx!=0): variables.append(new_name)

#plot all input features
def plot_features(trainDF,inp_feat):
    fig, ax = plt.subplots()
    print('Inside plotting funct\n',trainDF)
    for ft in inp_feat:
        print('Plotting feature ', ft)
        #pandas_trainDF = ak.to_pandas(trainDF)
        pandas_trainDF = trainDF
        trainDF_2P = pandas_trainDF.query("ljet_flavourlabel_0==0")
        trainDF_1P = pandas_trainDF.query("ljet_flavourlabel_0==1")
        max_val = int(max(trainDF_2P[ft]))
        min_val = int(min(trainDF_2P[ft]))
        number_of_bins = 100
        edges = (max_val-min_val)/number_of_bins
        bins = []
        i = 0
        while i<number_of_bins:
            bins.append(min_val+i*edges)
            i=i+1
        plt.hist(trainDF_2P[ft], histtype='step', label="2P", density=True, bins=bins)
        plt.hist(trainDF_1P[ft], histtype='step', label="1P", density=True, bins=bins)
        ax.legend()
        plt.xlabel(ft)
        plt.savefig('model_figs/'+ft+'.png')
        plt.cla()

#derive secondary features
def add_features(trainDF):
    trainDF['dphi_subjets']=trainDF['tjet_phi_0']-trainDF['tjet_phi_1']
    trainDF['deta_subjets']=trainDF['tjet_eta_0']-trainDF['tjet_eta_1']
    return trainDF

#features importance ranking
def get_feature_ranking(inp_feat,bst):
    features_map = {}
    for name,imp in zip(inp_feat, bst.feature_importances_):
        features_map[name]=float(imp)
    features_map = dict(sorted(features_map.items(), key=lambda item:item[1],
                        reverse=True))
    print('Features importance ', features_map)
    return features_map

#Calculate accuracy
def get_accuracy(bst,X_test,Y_test):
    y_pred = bst.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return 0

#Performance evaluation
def get_perf_plots(bst,X,Y,X_test, Y_test):
    results = bst.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.xlabel("Iteration")
    plt.title("XGBoost Log Loss")
    plt.savefig('model_figs/logloss.png')

    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.xlabel("Iteration")
    plt.title('XGBoost Classification Error')
    plt.savefig('model_figs/error.png')

    # plot classification auc
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.ylabel('AUC')
    plt.xlabel("Iteration")
    plt.title('XGBoost Training Performance')
    plt.savefig('model_figs/auc.png')

    # plot ROC curve
    y_pred_proba_train = bst.predict_proba(X)
    y_pred_proba_test  = bst.predict_proba(X_test)

    fpr_train, tpr_train, _ = sklm.roc_curve(Y, y_pred_proba_train[:,1])
    fpr_test, tpr_test, _   = sklm.roc_curve(Y_test, y_pred_proba_test[:,1])

    auc_train = sklm.auc(fpr_train, tpr_train)
    auc_test  = sklm.auc(fpr_test, tpr_test)

    fig, ax = plt.subplots()
    plt.title(f"ROC curve, AUC=(test: {auc_test:.4f}, train: {auc_train:.4f})")
    plt.plot(fpr_test, tpr_test, label="test data")
    plt.plot(fpr_train, tpr_train, label="train data")
    ax.legend()
    plt.ylabel('ROC Curve')
    plt.savefig('model_figs/roc.png')
    return 0

#survival probabilities
def get_class_probabilities_lab(bst, trainDF, inp_feat):
    trainDF_2P = trainDF[trainDF["ljet_flavourlabel_0"]==0]
    trainDF_1P  = trainDF[trainDF["ljet_flavourlabel_0"]==1]
    pred_proba_2P = bst.predict_proba(trainDF_2P[inp_feat])
    pred_proba_1P = bst.predict_proba(trainDF_1P[inp_feat])

    fig, ax = plt.subplots()
    plt.title('Training sample')
    number_of_bins = 100
    edges = 1/number_of_bins
    bins = []
    i = 0
    while i<number_of_bins:
        bins.append(i*edges)
        i=i+1
    plt.hist(pred_proba_2P[:,1], histtype='step', label="Two prong",bins=bins)
    plt.hist(pred_proba_1P[:,1], histtype='step', label="One prong",bins=bins)
    ax.legend()
    plt.xlabel('Two prong probability')
    plt.savefig('model_figs/labeled_prob.png')
    return 0

##unlabeled data
def get_class_probabilities_unlab(bst, inp_feat, variables, large_jet_vars):

    train_file = "../364703.root"
    testDF = get_file(train_file, variables, large_jet_vars)
    print("Test dataframe\n", testDF)

    y_pred = bst.predict(testDF[inp_feat])
    predictions = [round(value) for value in y_pred]
    test_proba = bst.predict_proba(testDF[inp_feat])

    fig, ax = plt.subplots()
    plt.title('Unlabeled sample')
    number_of_bins = 100
    edges = 1/number_of_bins
    bins = []
    i = 0
    while i<number_of_bins:
        bins.append(i*edges)
        i=i+1
    plt.hist(test_proba[:,1], histtype='step', label="Unlabeled",bins=bins)
    ax.legend()
    plt.xlabel('Two prong probability')
    plt.savefig('model_figs/unlabeled_prob.png')
    return 0

#main
if __name__ == '__main__':
    main()
