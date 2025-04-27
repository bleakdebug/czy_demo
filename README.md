# unKR: A Python Library for Uncertain Knowledge Graph Reasoning by Representation Learning
<p align="center">
    <a href="https://pypi.org/project/unKR/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/unKR">
    </a>
    <a href="https://github.com/seucoin/unKR/blob/main/LICENSE">
        <img alt="Pypi" src="https://img.shields.io/badge/license-Apache--2.0-yellowgreen">
    </a>
    <!-- <a href="">
        <img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-brightgreen">
    </a> -->
    <a href="https://seucoin.github.io/unKR/">
        <img alt="Documentation" src="https://img.shields.io/badge/Doc-online-blue">
    </a>
</p>

SAUR model is a complex model for uncertain knowledge graph inference, which combines BERT model, Graph Convolutional Network (GCN), Long Short Term Few Pages Network (LSTM), and was proposed by Zhao et al. (2025). And unKR is a UKG inference toolkit that has integrated nine UKG inference models proposed in recent years. It provides a unifying workflow to implement a variety of uncertain knowledge graph representation learning models to complete UKG reasoning. This project integrates the SAUR model into the unKR toolkit using [PyTorch Lightning](https://www.pytorchlightning.ai/) and pyTorch frameworks. The integrated SAUR model is trained on the data set provided by unKR, and the performance optimization of the SAUR model is realized.

<h3 align="center">
    <img src="pics/unKR.svg", width="1000">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/unKR.svg" width="1000"/></a>
<p> -->

<!-- ![demo](./pics/demo.gif) -->

<img src="pics/demo.gif">

<!-- <img src="pics/demo.gif" width="900" height="500" align=center> -->


## Datasets
unKR provides three public UKG datasets including CN15K, NL27K, and PPI5K. The following table shows the source, the number of entities, relations, and facts of each dataset.

| Dataset |   Source   | #Entity  | #Relation | #Fact |
|:-------:|:----------:|:---------:|:---------:|:-----------:|
|  CN15K  | ConceptNet |   15000   |    36     |   241158    |
|  NL27K  |    NELL    |   27221   |    404    |   175412    |
|  PPI5K  |   STRING   |   4999    |     7     |   271666    |
                                                                                                                                                 |

## Reproduced Results
unKR determines two tasks, confidence prediction and link prediction, to evaluate models' ability of UKG reasoning. For confidence prediction task, MSE (Mean Squared Error) and MAE (Mean Absolute Error) are reported. For link prediction task, Hits@k(k=1,3,10), MRR (Mean Reciprocal Rank), MR (Mean Rank) under both raw and filterd settings are reported. In addition, we choose high-confidence (>0.7) triples as the test data for link prediction.

Here are the reproduce results of SAUR model on NL27K dataset with unKR. 

### Confidence prediction
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Model</th>
            <th>MSE</th>
            <th>MAE </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="8">Normal</td>
            <td>SAUR</td>
            <td>0.08920 </td>
            <td>0.22194  </td>
        </tr>
        <tr>
            <td>PASSLEAF(ComplEx)</td>
            <td>0.02434 </td>
            <td>0.05176  </td>
        </tr>
    </tbody>
</table>

### Link prediction
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal</td>
            <td>SAUR</td>
            <td>0.156 </td>
            <td>0.385 </td>
            <td>0.543 </td>
            <td>0.299 </td>
            <td>488.051 </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.814 </td>
            <td>0.918 </td>
            <td>0.957 </td>
            <td>0.870 </td>
            <td>384.471 </td>
        </tr>
    </tbody>
</table>

<br>

## Usage

### Installation

**Step1** Create a virtual environment using ```Anaconda``` and enter it.

```bash
conda create -n unKR_SAUR python=3.9
conda activate unKR_SAUR
```

**Step2**  Install package.
+ Install from source
```bash
git clone https://github.com/bleakdebug/czy_demo.git
cd unKR_SAUR
pip install -r requirements.txt
python setup.py install
```

### Data Format
For SAUR model, `train.tsv`, `val.tsv`, and `test.tsv` are required. 

- `train.tsv`: All facts used for training in the format `(h, r, t, s)`, one fact per line.

- `val.tsv`: All facts used for validation in the format `(h, r, t, s)`, one fact per line.

- `test.tsv`: All facts used for testing in the format `(h, r, t, s)`, one fact per line.


### Parameter Setting
You can set up parameters by [config](https://github.com/bleakdebug/czy_demo/tree/main/config) file. 


### Model Training
```bash
python SAURdemo.py --load_config --config_path <your-config>
```

### Model Testing
```bash
python SAURdemo.py --test_only --checkpoint_dir <your-model-path>
```



