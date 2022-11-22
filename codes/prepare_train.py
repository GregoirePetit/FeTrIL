from configparser import ConfigParser
from multiprocessing import Pool
import os
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
    print("Preparing state",state_id, "of", il_states)
    root_path = os.path.join(feat_root,"fetril",dataset,"seed"+str(random_seed),"b"+str(first_batch_size),"t"+str(il_states),"train","batch"+str(state_id))

    nb_classes = first_batch_size + (state_id) * incr_batch_size
    def decompose_class(n):
        file_path = os.path.join(root_path, str(n))
        if os.path.exists(file_path):
            try:
                os.makedirs(file_path+'_decomposed', exist_ok=True)
                compteur = 0
                with open(file_path, 'r') as f:
                    for line in f:
                        with open(os.path.join(file_path+'_decomposed', str(compteur)), 'w') as f2:
                            f2.write(line)
                        compteur += 1
            except:
                pass

    with Pool() as p:
        p.map(decompose_class, range(nb_classes))