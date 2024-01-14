# T-GCN-PyTorch

This is a PyTorch implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320) for classification purpose. Original paper was only for linear regression purpose. So, two extra files ([supervised_linearClassification.py](tasks/supervised_linearClassification.py), [supervised_logistic.py](tasks/supervised_logistic.py)) have been added here. Moreover, [functions.py](utils/data/functions.py), [spatiotemporal_csv_data.py](utils/data/spatiotemporal_csv_data.py), [main.py](main.py) have been modified.

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch).

Notice that [the original implementation](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-TensorFlow) is in TensorFlow, which performs a tiny bit better than this implementation for now.

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training for Linear Regression

```bash
# GCN
python main.py --model_name GCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# GRU
python main.py --model_name GRU --max_epochs 3000 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# T-GCN
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.

## Model Training for Classification
First change the [__init__.py](tasks/__init__.py) accordingly.

For configuration
```bash
--model_name TGCN --data flu_13_14 --batch_size 64 --hidden_dim 64 --max_epochs 300 --learning_rate 0.001 --seq_len 5 --pre_len 1 --lead_time 0 --num_class 5 --split_ratio 0.5
```

## Model Training to Compare with DETECTIVE
For the sake of fairness, when comparing with DETECTIVE, we followed the following steps:
- First replace the [main](main.py) and [supervised_linearClassification](tasks/supervised_linearClassification.py) file with [main](DETECTIVE_Comparison/main.py) and [supervised_linearClassification](DETECTIVE_Comparison/supervised_linearClassification.py). 
-For configuration:
    - Civil Unrest
```bash
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data argentina --adjust_feat 1249 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data brazil --adjust_feat 1288 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data chile --adjust_feat 1214 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data colombia --adjust_feat 1230 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data mexico --adjust_feat 1262 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data paraguay --adjust_feat 1105 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data uruguay --adjust_feat 1149 --lead_time 0 --num_class 3 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data venezuela --adjust_feat 1251 --lead_time 0 --num_class 3 --split_ratio 0.5

--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data china_air --adjust_feat 1343 --lead_time 0 --num_class 4 --split_ratio 0.5

--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data flu_11_12 --adjust_feat 524 --lead_time 0 --num_class 5 --split_ratio 0.5
--model_name TGCN  --batch_size 64 --hidden_dim 64 --max_epochs 480 --learning_rate 0.01 --seq_len 1 --pre_len 1 --data flu_13_14 --adjust_feat 524 --lead_time 0 --num_class 5 --split_ratio 0.5

```
