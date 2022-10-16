from configparser import ConfigParser
import sys
import os
import numpy as np
from sklearn.preprocessing import Normalizer
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

test_feats_path = os.path.join(feat_root,dataset,"b"+str(first_batch_size),"test/")
svms_dir = os.path.join(classifiers_root,"fetril",dataset,"b"+str(first_batch_size),"t"+str(il_states))
pred_path = os.path.join(pred_root,"fetril",dataset,"b"+str(first_batch_size),"t"+str(il_states))
model_dir = svms_dir
os.makedirs(pred_path, exist_ok=True)
T = il_states


def compute_feature(i):
   corresponding_batch = (i-first_batch_size)//((nb_classes-first_batch_size)//T)+1
   if i<first_batch_size:
      corresponding_batch=0
   test_feats = os.path.join(test_feats_path,str(i))
   for batchs in range(corresponding_batch, T+1):
      os.makedirs(os.path.join(pred_path,"batch"+str(batchs)),exist_ok=True)
      pred_file = os.path.join(pred_path,"batch"+str(batchs),str(i))
      if not os.path.exists(pred_file):
         with open(pred_file, "w") as f_pred:
            syns = []
            f_list_syn = list(range(((nb_classes-first_batch_size)//T)*(batchs)+first_batch_size))
            for syn in f_list_syn:
               syn = str(syn)
               syns.append(syn)
            weights_list = []  
            biases_list = []
            for syn in range(len(syns)):
               line_cnt = 0
               target_model = os.path.join(model_dir,"batch"+str(batchs),str(syn)+".model")
               f_model = open(target_model)
               for line in f_model:
                  line = line.rstrip()
                  if line_cnt == 0:
                     parts = line.split(" ")
                     parts_float = []
                     for pp in parts:
                        parts_float.append(float(pp))
                     weights_list.append(parts_float)
                  elif line_cnt == 1:
                     biases_list.append(float(line))
                  line_cnt = line_cnt + 1
               f_model.close()
            f_test_feat = open(test_feats, 'r')
            for vline in f_test_feat.readlines():
               vparts = vline.split(" ")
               crt_feat = [[float(vp) for vp in vparts]]
               crt_feat = Normalizer().fit_transform(crt_feat)[0]
               pred_dict = []
               for cls_cnt in range(len(weights_list)):
                  cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
                  pred_dict.append(-cls_score)
               pred_line = ""
               predictions_idx = sorted(range(len(pred_dict)), key=lambda k: -pred_dict[k])
               for idx in predictions_idx:
                  pred_line = pred_line+" "+str(idx)+":"+str(pred_dict[idx]) 
               pred_line = pred_line.lstrip()
               f_pred.write(pred_line+"\n")
            f_test_feat.close()
      else:
         print("exists predictions file:",pred_file)
with Pool() as p:
   p.map(compute_feature, range(nb_classes))
