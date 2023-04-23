# TransFOL

![ ](figure/3-2.png)

**This is the data and code for our paper** `TransFOL: A Logical Query Model Based on Cross-Transformer for Drug–Drug Interaction Prediction`.

## Prerequisites

* `cuda version < 11.0`
* `pytorch>=1.7.1 & <=1.9`
  * Note: Pytorch version greater than 1.9 has OOM bugs. See <https://github.com/pytorch/pytorch/issues/67680>.
* `torch-geometric`

Example installation using [`conda`](https://conda.io):

```bash
# Use the cuda version that matches your nvidia driver and pytorch
conda install pytorch=1.8.1 cudatoolkit=10.1 pyg -c pyg -c pytorch -y
```

## Datastes

We provide the dataset in the [data](data/) folder.

| Data | Source | Description |
| --- | --- | --- |
| [Drugbank](data/drugbank/)| [This link](https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/data/) | A drug-drug interaction network betweeen 1,709 drugs with 136,351 interactions. |
| [TWOSIDES](data/TWOSIDES/) | [This link](http://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html) | A drug-drug interaction network betweeen 645 drugs with 46,221 interactions. |
| [DrugCombDB](data/DrugCombDB) | [This link](http://drugcombdb.denglab.org/) | An integrated biomedical network contains more than 4 million drugs and other entities. We extract a subgraph dataset containing the relationship "*drug-dose-target-response*" from it.|
| [Phenomebrowser](data/Phenomebrowser) | [This link](http://www.phenomebrowser.net/#/) | An integrated biomedical network include  61,146 medical entities and 1,246,726 medical entity pairs.  We extract sub-datasets that contain the "*mammal symptom-human symptom-drug-disease*" relationship from it. |

## Reproduction

The parameters in the paper is preloaded in [`./src/configs/`](src/configs/).
Change `root_dir` option for the location to save model checkpoints.

The location for the extracted dataset
should be specified in the `data_dir` in the config files.
For exmpale, if the `drugbank` dataset is in `/data/drugbank`,
this is what the `data_dir` options should be set.

Alternatively, pretrained models are available
at [Google Drive](https://drive.google.com/drive/folders/1FI4TqRI_bXUSNMAuf0F2qUCmUQGjtn_j).

To reproduce all results for `drugbank`:

```bash
transfol="python main.py -c configs/drugbank.json"
$ transfol training_2i 
$ transfol testing_2i
```

## Documentation

```
src
  │  data_util.py
  │  deter_util.py
  │  graph_util.py
  │  main.py
  │  metric.py
  │  model.py
  │  sampler.py
  │  train.py
  │
  ├─configs
    │      drugbank.json
    │      DrugCombDB.json
    │      Phenomebrowser.json
    │      TWOSIDES.json
    │
  ├─pretrained_model
  └─tasks
          base.py
          betae.py
          pretrain.py
          real_query.py
          reasoning.py
          __init__.py
```

### Model

The basic structure of our model an be found in `model.py`.
The model can be divided into 4 parts, triplet transform, enhancement module, Cross-Transformer and GCN decoder. They can be used in function `TokenEmbedding`, `Cross_Transformer` and `GCN`.

### Training

Training-related utilities can be found in [`train.py`](./src/train.py).
They accept `Iterator`'s that yield batched data,
identical to the output of a `torch.utils.data.DataLoader`.
The most useful functions are `main_mp()` and `ft_test()`.

`TrainClient` scatters data onto different workers
and perform multi-GPU training based on `torch.nn.parallel.DistributedDataParallel`.

### Config Files

Each config file is a JSON key-value mapping that maps a task name to a task.
The tasks can be run directly from the command line:

```bash
python main.py <task_name> [<task_name>...]
```

In a specific task, `base` option specifies the task it should inherit from.
`type` option specifies the type of operation of this configuration.
See [`./src/main.py`](./src/main.py) for a full list of available options.

## Troubleshooting

<details>

<summary>CUDA Out of Memory</summary>

We run experiments with V100(32GB) GPU, please reduce the batch size if you don't have enough resources. Be aware that smaller batch size will hurt the performance for contrastive training
If the issue persists after adjusting batch size, downgrade pytorch to as early as possible (e.g. LTS 1.8.1 as of 2021/03).
This is possibly due to memory issues in higher pytorch versions.
See <https://github.com/pytorch/pytorch/issues/67680> for more information.

</details>

<details>

<summary>torch-geometric installation is failed</summary>

Please try downgrading the cuda version. Due to library dependency, torch_cluster, torch_scatter, torch_sparse and torch_spline_conv are required to install torch-geometric installations.

</details>

## Authors

**Junkai Cheng** @github.com/Cheng0829 \
**Email:** chengjunkai829@gmail.com & Cheng0829@dlmu.edu.cn \
**Site:** [GitHub](https://github.com/Cheng0829)

**Yijia Zhang** @github.com/ItrecTextming \
**Email:** zhangyijia@dlmu.edu.cn \
**Site:** [GitHub](https://github.com/ItrecTextming)

## Acknowledgement

We refer to the code of [kgTransformer](https://github.com/THUDM/kgTransformer). Thanks for their contributions.