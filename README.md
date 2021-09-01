# Few-shot NER
Code for paper: Few-Shot Named Entity Recognition: An Empirical Baseline Study
https://arxiv.org/pdf/2012.14978.pdf

## Dependencies:

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

## Quickstart

Download our models pre-trained on WiFine (Wikipedia) to ```src/pretrained_models/``` from [here](https://drive.google.com/drive/folders/1IkilP648x2aGVY1odo_NEDt7stTT5Z1H?usp=sharing).
To load model pre-trained on WiFine (Wikipedia) and fine-tune on CONLL2003 dataset, 
```
cd src
bash ./r_naiveft.sh
```
By default, this runs 10 rounds of experiments with different sets of 5-shot seeds and allows self-training on the whole dataset.

### Multiple Runs

To run multiple rounds of experiments on various few-shot seeds (e.g., 10 rounds), set
```
--train_text few_shot_5 --train_ner few_shot_5 --few_shot_sets 10
```
in the command. ''few_shot_5'' is the common file name of the seed files. The average results of F1-score will be output at the end.

If only one round is needed, you need to set the complete file names for training 
```
--train_text train.words --train_ner train.ner 
```

### Allow Self-training

Set the files for self-training by
```
--unsup_text train.words --unsup_ner train.ner
```
The labels in ''unsup_ner'' are not used in training, but will be used for evaluation before self-training to give you a hint on how much potential you can get from self-training.

To disallow self-training, just remove the two relevant flags.

### Use Your Own Pre-trained Model

If you want to load your own pre-trained model, set
```
--load_model True --load_model_name path/to/your/model
```
If you want to load the original pre-trained Roberta model (https://arxiv.org/abs/1907.11692), set
```
--load_model False
```

### Use Prototype-based Methods

You can use prototype-based methods by running the following command
```
bash ./r_proto.sh
```
In this script, you can also allow or disallow multiple runs, and customize pre-trained models.

## Benchmark Datasets

In our paper, we studied the result on 10 benchmark datasets. For the public ones, we provide our few-shot seed sets and the whole dataset [here](https://drive.google.com/drive/folders/1CUTXJzhV1FvLjhA-gQtsofr7JBCEfP-A?usp=sharing). For the other datasets which require license for access, if you want the same set of few-shot seeds, please first get the license for the whole dataset and then ask the first author for the sampled few-shot seeds.

Dataset | Domain | Included [here](https://drive.google.com/drive/folders/1CUTXJzhV1FvLjhA-gQtsofr7JBCEfP-A?usp=sharing)
--- | --- | --- 
CoNLL | News | :heavy_check_mark:
Onto | General | ✖️
WikiGold | General | :heavy_check_mark:
WNUT17 | Social Media | :heavy_check_mark:
MITMovie | Review | :heavy_check_mark:
MITRestaurant | Review | :heavy_check_mark:
SNIPS | Dialogue | :heavy_check_mark:
ATIS | Dialogue | :heavy_check_mark:
Multiwoz | Dialogue | :heavy_check_mark:
i2b2 | Medical | ✖️
