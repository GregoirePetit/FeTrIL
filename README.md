# FeTrIL
This is the official code for FeTrIL (WACV2022): Feature Translation for Exemplar-Free Class-Incremental Learning

```
python codes/compute_distances.py configs/cifar100_b50_t10.cf
python codes/prepare_train.py configs/cifar100_b50_t10.cf
python codes/train_classifiers.py configs/cifar100_b50_t10.cf
python codes/clean_train.py configs/cifar100_b50_t10.cf
python codes/compute_predictions.py configs/cifar100_b50_t10.cf
python codes/eval.py configs/cifar100_b50_t10.cf
```
