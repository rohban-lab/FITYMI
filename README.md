# Fake It Till You Make It: Towards Accurate Near-Distribution Novelty Detection
Official PyTorch implementation of ["Fake It Till You Make It: Towards Accurate Near-Distribution Novelty Detection" ](https://arxiv.org/abs/2205.14297) by
Hossein Mirzaei,
 [Mohammadreza Salehi](https://scholar.google.com/citations?user=kpT3gcsAAAAJ&hl=en),
 Sajjad Shahabi,
  [Efstratios Gavves](https://scholar.google.com/citations?user=QqfCvsgAAAAJ&hl=en), 
  [Cees G. M. Snoek](https://scholar.google.com/citations?user=0uKdbscAAAAJ&hl=en), 
  [Mohammad Sabokrou](https://scholar.google.com/citations?user=jqHXvT0AAAAJ&hl=en), 
  [Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en)

<p align="center">
<img src="https://user-images.githubusercontent.com/33581331/170314540-1689d686-1d53-43e2-bd43-253b51f1b805.png" alt="Main method overview" height="550" width="750"/>
</p>

## 1. Requirements
### Environment
The current version requires the following python and CUDA versions:
- python 3.7+
- CUDA 11.1+

Additionally, the list of the packages used for this implementation is available in the `requirements.txt` file. To install them, use the following command:
```
pip install -r requirements.txt
```

### Datasets 
To replicate the results of the experiments, please download the following generated anomaly datasets:
- [CIFAR-10 Generated Dataset](https://drive.google.com/file/d/1jMKZPqFTldsO80U3o79KDTSagUVUAl-5)
- [CIFAR-100 Generated Dataset](https://drive.google.com/file/d/1-1-L4qWCTg08lBfPF9qDBMYOaHne6mR5)

Each of these datasets is used as counterfeit anomalies during the model's training.
Each dataset contains generated samples for every class of the dataset. These samples are concatenated together with respect to their class number.
E.g., the first 5000 images of the `cifar10_training_gen_data.npy` are generated samples based on the first class of the CIFAR-10 dataset (Airplane), the second 5000 images are based on the second label, and so on. (Note that for the CIFAR-100 dataset number of samples for each class is 2500).

Currently, only the generated samples for these two datasets are available. For other datasets, please refer to this 
[implementation](https://github.com/yang-song/score_sde_pytorch)
of an SDE-based generative model, and follow the guidelines provided in the paper to generate anomaly samples.

## 2. Training and Evaluation
### One-Class Novelty Detectiion
```
python main.py --dataset <DATASET> --label <NORMAL_CLASS> --output_dir <RESULTS_DIR> --normal_data_path <NORMAL_DATA_DIR>\
    --gen_data_path <GEN_DATA_DIR> --pretrained_path <MODEL_DIR> --train_batch_size 16 --eval_batch_size 16 --nnd --download_dataset
```
> The option --label indicates the normal class.
> Use the --gen_data_path option to set the path to generated datasets provided in the [Datasets](#datasets) section.
> The --pretrained_path option specifies the path to the pre-trained model. Please refer to this 
> [implementation](https://github.com/jeonsworld/ViT-pytorch)
> of the ViT model to see the list of the available models and how to access them.
> Finally, the --nnd option should be used to evaluate the model on the NND setting described in the paper. This option is only available for the CIFAR-10 dataset.

An example of training and evaluation of the model on the first class of both datasets is available in this
[notebook](https://colab.research.google.com/drive/1nrYT6cfNjKnBVc7wgYemLwqNByDb9Coi).

For high-resolution datasets, it is recommended to increase the learning rate to 5e-3 and the epoch count to 150.

## 3. Results
The model's performance on the available datasets is provided in the table below:
|       | CIFAR-10 (ND) | CIFAR-10 (NND) | CIFAR-100 |
|-------|:-------------:|:--------------:|:---------:|
| AUROC |      99.1     |      90.0      |    98.1   |

To see the results on other datasets, please refer to our paper.

## 4. CIFAR-10-FSDE
In this section, the dataset for the CIFAR-10-FSDE benchmark is provided. This dataset is a subsample of the generated data on the CIFAR-10 dataset. It can be used as a measure to evaluate anomaly detection and out-of-distribution methods in the near-distribution setting.

To download the dataset, please use the link provided below:
- [CIFAR-10-FSDE](https://drive.google.com/file/d/1_sKwq1yG-0zdvUHBRXItJ6Cib6lpQJja)

This dataset contains the test samples for every class of the CIFAR-10 dataset (1000 samples per class). These samples are concatenated together in the manner described in the [Datasets](#Datasets) section.

To see the performance of various anomaly detection and out-of-distribution methods, please refer to our paper.

## 5. Citation
If you find this useful for your research, please cite the following paper:
``` bash
@misc{https://doi.org/10.48550/arxiv.2205.14297,
  doi = {10.48550/ARXIV.2205.14297},
  url = {https://arxiv.org/abs/2205.14297},
  author = {Mirzaei,  Hossein and Salehi,  Mohammadreza and Shahabi,  Sajjad and Gavves,  Efstratios and Snoek,  Cees G. M. and Sabokrou,  Mohammad and Rohban,  Mohammad Hossein},
  keywords = {Computer Vision and Pattern Recognition (cs.CV),  Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Fake It Till You Make It: Towards Accurate Near-Distribution Novelty Detection},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
