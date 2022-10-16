from configparser import ConfigParser
import os
import sys
import csv
import numpy as np
from multiprocessing import Pool

if len(sys.argv) != 2:
    print('Arguments: config')
    sys.exit(-1)

cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)

cp = cp['config']
nb_classes = int(cp['nb_classes'])
dataset = cp['dataset']
first_batch_size = int(cp["first_batch_size"])
il_states = int(cp["il_states"])
feat_root = cp["feat_root"]
pred_root = cp["pred_root"]
classifiers_root = cp["classifiers_root"]
t = il_states
root_path_pred = os.path.join(pred_root,"fetril",dataset,"b"+str(first_batch_size),"t"+str(il_states))

batch_size = (nb_classes-first_batch_size)//il_states

batches = range(t+1)
resultats = {}
def flatten(t):
    return [item for sublist in t for item in sublist]
def compute_score(nb_batch):
    path_pred = os.path.join(root_path_pred,"batch"+str(nb_batch))
    y_pred = []
    score_top5 = []
    y_true = []
    for c in range(first_batch_size+batch_size*nb_batch):
        with open(os.path.join(path_pred,str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])
    y_pred = np.asarray(flatten(y_pred))
    y_pred_top5 = flatten(score_top5)
    y_true = np.asarray(flatten(y_true))
    return((nb_batch,[np.mean(y_pred == y_true),np.mean(y_pred_top5)]))

def detailled_score(nb_batch):
    path_pred = os.path.join(root_path_pred,"batch"+str(nb_batch))
    res = []
    for c in range(first_batch_size+batch_size*nb_batch):
        y_pred = []
        score_top5 = []
        y_true = []
        with open(os.path.join(path_pred,str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])
        y_pred = np.asarray(flatten(y_pred))
        y_pred_top5 = flatten(score_top5)
        y_true = np.asarray(flatten(y_true))
        prdic = np.mean(y_pred == y_true)
        res.append(prdic)

    return([np.mean(res[:first_batch_size])]+[np.mean(res[i:i+batch_size]) for i in range(first_batch_size,len(res),batch_size)])

with Pool() as p:
    resultats = dict(p.map(compute_score, batches))
top1=[]
top5=[]
for batch_number in batches:
    top1.append(resultats[batch_number][0])
    top5.append(resultats[batch_number][1])
print("top1:",[round(100*elt,2) for elt in top1])
print("top5:",[round(100*elt,2) for elt in top5])
print(f'top1 = {sum(top1)/len(top1):.3f}, top5 = {sum(top5)/len(top5):.3f}')
