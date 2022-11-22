# FeTrIL
This is the official code for [FeTrIL](https://gregoirepetit.github.io/projects/FeTrIL) (WACV2022): Feature Translation for Exemplar-Free Class-Incremental Learning

<p align="center">
<img src="medias/teaser.png" />
</p>

## Abstract

## Installation

To install the required package, please run the following command (conda is required):

```bash
conda env create -f fetril.yml
```

## Usage

### Configuration

Using the [configs/cifar100_b50_t10.cf](https://github.com/GregoirePetit/FeTrIL/blob/main/configs/cifar100_b50_t10.cf) file, you can prepare your experiment. You can change the following parameters:
- `nb_classes`: number of classes in the dataset
- `dataset`: name of the dataset (for exemple: cifar100, tinyimagenet, etc.)
- `first_batch_size`: number of classes in the first state
- `il_states`: number of incremental states (note that the first state is not counted as incremental)
- `random_seed`: random seed for the experiment (default: -1 will not shuffle the class order)
- `num_workers`: number of workers for the dataloader
- `regul`: regularization parameter for the classifier
- `toler`: tolerance parameter for the classifier
- `epochs_lucir`: number of epochs for the LUCIR part of the training
- `epochs_augmix_ft`: number of epochs for the AugMix fine-tuning part of the training
- `list_root`: path to the directory where the lists of images and classes are stored
- `model_root:`: path to the directory where the models will be saved
- `feat_root`: path to the directory where the features will be saved
- `classifiers_root`: path to the directory where the classifiers will be saved
- `pred_root`: path to the directory where the predictions will be saved
- `mean_std`: path to the file containing the mean and std of the dataset

### Experiments

Once the configuration file is ready, you can run the following command to launch the experiment:

#### Compute the pseudo-features according to the FeTrIL method:
```bash
python codes/compute_distances.py configs/cifar100_b50_t10.cf
```

#### Format the pseudo-features to be used by the dataloader:
```bash
python codes/prepare_train.py configs/cifar100_b50_t10.cf
```

#### Train the classifiers:
```bash
python codes/train_classifiers.py configs/cifar100_b50_t10.cf
```

#### Clean the pseudo-features used by the dataloader to train the classifiers:
```bash
python codes/clean_train.py configs/cifar100_b50_t10.cf
```

#### Compute the predictions on the test set:
```bash
python codes/compute_predictions.py configs/cifar100_b50_t10.cf
```

#### Compute the accuracy on the test set:
```bash
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
