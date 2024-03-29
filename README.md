# FeTrIL
This is the official code for [FeTrIL](https://gregoirepetit.github.io/projects/FeTrIL) (WACV2023): Feature Translation for Exemplar-Free Class-Incremental Learning

<p align="center">
<img src="medias/teaser.png" />
</p>

## Abstract

Exemplar-free class-incremental learning is very challenging due to the negative effect of catastrophic forgetting. A balance between stability and plasticity of the incremental process is needed in order to obtain good accuracy for past as well as new classes. Existing exemplar-free class-incremental methods focus either on successive fine tuning of the model, thus favoring plasticity, or on using a feature extractor fixed after the initial incremental state, thus favoring stability. We introduce a method which combines a fixed feature extractor and a pseudo-features generator to improve the stability-plasticity balance. The generator uses a simple yet effective geometric translation of new class features to create representations of past classes, made of pseudo-features. The translation of features only requires the storage of the centroid representations of past classes to produce their pseudo-features. Actual features of new classes and pseudo-features of past classes are fed into a linear classifier which is trained incrementally to discriminate between all classes. The incremental process is much faster with the proposed method compared to mainstream ones which update the entire deep model. Experiments are performed with three challenging datasets, and different incremental settings. A comparison with ten existing methods shows that our method outperforms the others in most cases.

## Results

For the following results *T* denominates the number of incremental states. The first, non-incremental, state contains half of the classes, excepted the *T=20* and *T=60* setups for the 100-classes datasets, where it accounts for 40 classes. Whenever available, results of compared methods marked with ∗ are reproduced either from their initial paper or from [SSRE](https://arxiv.org/abs/2203.06359) for EFCIL. The other results are recomputed using the original configurations of the methods.

### CIFAR-100
| **CIL Method** | *T=5* | *T=10* | *T=20* | *T=60* |
|----------------|-------|--------|--------|--------|
| [EWC*](https://www.pnas.org/doi/10.1073/pnas.1611835114) (PNAS'17) | 24.5 | 21.2 | 15.9 | x |
| [LwF-MC*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) (CVPR'17) | 45.9 | 27.4 | 20.1 | x | 29.1 | 23.1 | 17.4 | x | - | 31.2 | - | x | - | - | - |
| [DeeSIL](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Belouadah_DeeSIL_Deep-Shallow_Incremental_Learning._ECCVW_2018_paper.pdf) (ECCVW'18) | 60.0 | 50.6 | 38.1 | x | 49.8 | 43.9 | 34.1 | x | 67.9 | 60.1 | 50.5 | x | 61.9 | 54.6 | 45.8 |
| [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) (CVPR'19) | 51.2 | 41.1 | 25.2 | x | 41.7 | 28.1 | 18.9 | x | 56.8 | 41.4 | 28.5 | x | 47.4 | 37.2 | 26.6 |
| [MUC*](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710698.pdf) (ECCV'20) | 49.4 | 30.2 | 21.3 | x | 32.6 | 26.6 | 21.9 | x | - | 35.1 | - | x | - | - | - |
| [SDC*](https://arxiv.org/abs/2004.00440) (CVPR'20) | 56.8 | 57.0 | 58.9 | x | - | - | - | x | - | 61.2 | - | x | - | - | - |
| [ABD*](https://arxiv.org/abs/2106.09701) (ICCV'21) | 63.8 | 62.5 | 57.4 | x | - | - | - | x | - | - | - | x | - | - | - |
| [PASS*](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf) (CVPR'21) | 63.5 | 61.8 | 58.1 | x | 49.6 | 47.3 | 42.1 | x | 64.4 | 61.8 | 51.3 | x | - | - | - |
| [IL2A*](https://proceedings.neurips.cc/paper/2021/hash/77ee3bc58ce560b86c2b59363281e914-Abstract.html) (NeurIPS'21) | <ins>66.0</ins> | 60.3 | 57.9 | x | 47.3 | 44.7 | 40.0 | x | - | - | - | x | - | - | - |
| [SSRE*](https://arxiv.org/abs/2203.06359) (CVPR'22) | 65.9 | <ins>65.0</ins> | **61.7** | x | 50.4 | 48.9 | 48.2 | x | - | 67.7 | - | x | - | - | - |
| FeTrIL<sup>fc</sup> (WACV'23)| 64.7 | 63.4 | 57.4 | <ins>50.8</ins> | <ins>52.9</ins> | <ins>51.7</ins> | <ins>49.7</ins> | <ins>41.9</ins> | <ins>69.6</ins> | <ins>68.9</ins> | <ins>62.5</ins> | <ins>58.9</ins> | <ins>65.6</ins> | <ins>64.4</ins> | <ins>63.4</ins> |
| FeTrIL (WACV'23) | **66.3** | **65.2** | <ins>61.5</ins> | **59.8** | **54.8** | **53.1** | **52.2** | **50.2** | **72.2** | **71.2** | **67.1** | **65.4** | **66.1** | **65.0** | **63.8** |

Average top-1 incremental accuracy in EFCIL with different numbers of incremental steps. FeTrIL results are reported with pseudo-features translated from the most similar new class.	"-" cells indicate that results were not available. "x" cells indicate that the configuration is impossible for that method. 


### Tiny-ImageNet
| **CIL Method** | *T=5* | *T=10* | *T=20* | *T=100* |
|----------------|-------|--------|--------|--------|
| [EWC*](https://www.pnas.org/doi/10.1073/pnas.1611835114) (PNAS'17) | 18.8 | 15.8 | 12.4 | x | - | 20.4 | - | x | - | - | - |
| [LwF-MC*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) (CVPR'17)| 29.1 | 23.1 | 17.4 | x | - | 31.2 | - | x | - | - | - |
| [DeeSIL](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Belouadah_DeeSIL_Deep-Shallow_Incremental_Learning._ECCVW_2018_paper.pdf) (ECCVW'18) | 49.8 | 43.9 | 34.1 | x | 67.9 | 60.1 | 50.5 | x | 61.9 | 54.6 | 45.8 |
| [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) (CVPR'19)| 41.7 | 28.1 | 18.9 | x | 56.8 | 41.4 | 28.5 | x | 47.4 | 37.2 | 26.6 |
| [MUC*](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710698.pdf) (ECCV'20) | 32.6 | 26.6 | 21.9 | x | - | 35.1 | - | x | - | - | - |
| [PASS*](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf) (CVPR'21) | 49.6 | 47.3 | 42.1 | x | 64.4 | 61.8 | 51.3 | x | - | - | - |
| [IL2A*](https://proceedings.neurips.cc/paper/2021/hash/77ee3bc58ce560b86c2b59363281e914-Abstract.html) (NeurIPS'21) | 47.3 | 44.7 | 40.0 | x | - | - | - | x | - | - | - |
| [SSRE*](https://arxiv.org/abs/2203.06359) (CVPR'22) | 50.4 | 48.9 | 48.2 | x | - | 67.7 | - | x | - | - | - |
| FeTrIL<sup>fc</sup> (WACV'23)| <ins>52.9</ins> | <ins>51.7</ins> | <ins>49.7</ins> | <ins>41.9</ins> | <ins>69.6</ins> | <ins>68.9</ins> | <ins>62.5</ins> | <ins>58.9</ins> | <ins>65.6</ins> | <ins>64.4</ins> | <ins>63.4</ins> |
| FeTrIL (WACV'23)| **54.8** | **53.1** | **52.2** | **50.2** | **72.2** | **71.2** | **67.1** | **65.4** | **66.1** | **65.0** | **63.8** |

Average top-1 incremental accuracy in EFCIL with different numbers of incremental steps. FeTrIL results are reported with pseudo-features translated from the most similar new class.	"-" cells indicate that results were not available. "x" cells indicate that the configuration is impossible for that method. 

### ImageNet-Subset
| **CIL Method** | *T=5* | *T=10* | *T=20* | *T=60* |
|----------------|-------|--------|--------|--------|
| [EWC*](https://www.pnas.org/doi/10.1073/pnas.1611835114) (PNAS'17) | - | 20.4 | - | x |
| [LwF-MC*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) (CVPR'17)| - | 31.2 | - | x |
| [DeeSIL](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Belouadah_DeeSIL_Deep-Shallow_Incremental_Learning._ECCVW_2018_paper.pdf) (ECCVW'18) | 67.9 | 60.1 | 50.5 | x |
| [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) (CVPR'19) | 56.8 | 41.4 | 28.5 | x |
| [MUC*](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710698.pdf) (ECCV'20)| - | 35.1 | - | x |
| [SDC*](https://arxiv.org/abs/2004.00440) (CVPR'20) | - | 61.2 | - | x |
| [PASS*](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf) (CVPR'21) | 64.4 | 61.8 | 51.3 | x |
| [SSRE*](https://arxiv.org/abs/2203.06359) (CVPR'22) | - | 67.7 | - | x |
| FeTrIL<sup>fc</sup> (WACV'23) | <ins>69.6</ins> | <ins>68.9</ins> | <ins>62.5</ins> | <ins>58.9</ins> |
| FeTrIL (WACV'23)| **72.2** | **71.2** | **67.1** | **65.4** |

Average top-1 incremental accuracy in EFCIL with different numbers of incremental steps. FeTrIL results are reported with pseudo-features translated from the most similar new class.	"-" cells indicate that results were not available. "x" cells indicate that the configuration is impossible for that method. 


### ImageNet
| **CIL Method** | *T=5* | *T=10* | *T=20* |
|----------------|-------|--------|--------|
| [DeeSIL](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Belouadah_DeeSIL_Deep-Shallow_Incremental_Learning._ECCVW_2018_paper.pdf) (ECCVW'18) | 61.9 | 54.6 | 45.8 |
| [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) (CVPR'19)  | 47.4 | 37.2 | 26.6 |
| FeTrIL<sup>fc</sup> (WACV'23) | <ins>65.6</ins> | <ins>64.4</ins> | <ins>63.4</ins> |
| FeTrIL (WACV'23)| **66.1** | **65.0** | **63.8** |

Average top-1 incremental accuracy in EFCIL with different numbers of incremental steps. FeTrIL results are reported with pseudo-features translated from the most similar new class.	"-" cells indicate that results were not available. "x" cells indicate that the configuration is impossible for that method. 






## Installation

### Environment

To install the required packages, please run the following command (conda is required), using [fetril.yml](fetril.yml) file:

```bash
conda env create -f fetril.yml
```

If the installation fails, please try to install the packages manually with the following command:

```bash
conda create -n fetril python=3.7
conda activate fetril
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install typing-extensions --upgrade
conda install pandas
pip install -U scikit-learn scipy matplotlib
```

### Dependencies

The code depends on the repository [utilsCIL](https://github.com/GregoirePetit/utilsCIL) which contains the code for the datasets and the incremental learning process. Please clone the repository on your home ([FeTrIL code](https://github.com/GregoirePetit/FeTrIL/blob/main/codes/scratch.py#L19) will find it) or add it to your PYTHONPATH:

```bash
git clone git@github.com:GregoirePetit/utilsCIL.git
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

#### Train the first model and extract the features:
```bash
python codes/scratch.py configs/cifar100_b50_t10.cf
```

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

```BibTeX
@InProceedings{Petit_2023_WACV,
    author    = {Petit, Gr\'egoire and Popescu, Adrian and Schindler, Hugo and Picard, David and Delezoide, Bertrand},
    title     = {FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3911-3920}
}
```
