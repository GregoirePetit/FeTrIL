from configparser import ConfigParser
from multiprocessing import Pool
import os
import shutil
import sys

if len(sys.argv) != 2:
    print('Arguments: config')
    sys.exit(-1)

cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)
cp = cp['config']
nb_classes = int(cp['nb_classes'])
dataset = cp['dataset']
random_seed = int(cp["random_seed"])
first_batch_size = int(cp["first_batch_size"])
il_states = int(cp["il_states"])
feat_root = cp["feat_root"]
incr_batch_size = (nb_classes-first_batch_size)//il_states

for state_id in range(il_states+1):
    print("Cleaning state",state_id, "of", il_states)
    root_path = os.path.join(feat_root,"fetril",dataset,"seed"+str(random_seed),"b"+str(first_batch_size),"t"+str(il_states),"train","batch"+str(state_id))

    nb_classes = first_batch_size + (state_id) * incr_batch_size
    def decompose_class(n):
        file_path = os.path.join(root_path, str(n))
        try:
            shutil.rmtree(file_path+'_decomposed')
        except:
            pass

    with Pool() as p:
        p.map(decompose_class, range(nb_classes))