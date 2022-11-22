# FeTrIL
This is the official code for [FeTrIL](https://gregoirepetit.github.io/projects/FeTrIL) (WACV2022): Feature Translation for Exemplar-Free Class-Incremental Learning

<p align="center">
<img src="medias/teaser.png" />
</p>

```
python codes/compute_distances.py configs/cifar100_b50_t10.cf
python codes/prepare_train.py configs/cifar100_b50_t10.cf
python codes/train_classifiers.py configs/cifar100_b50_t10.cf
python codes/clean_train.py configs/cifar100_b50_t10.cf
python codes/compute_predictions.py configs/cifar100_b50_t10.cf
python codes/eval.py configs/cifar100_b50_t10.cf
```

## Citation
If you find this code useful for your research, please cite our paper:
```
@article{petit2023fetril, 
 Title = {FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning}, 
 Author = {G. Petit, A. Popescu, H. Schindler, D. Picard, B. Delezoide}, 
 Journal = {Winter Conference on Applications of Computer Vision (WACV)}, 
 Year = {2023}
}
```
