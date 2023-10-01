# Pand-GNN

## This is our PyTorch implementation code for our paper: Pand-GNN: Unifying Positive and Negative Edges in Graph Neural Networks for Recommendation

Note. In this implementation, we use a new LightGIN as the GNN model in our Pand-GNN method. 


## Environment Requirements

The code has been tested under Python 3.7.7. The required packages are as follows:

* Pytorch == 1.5.0
* Pytorch Geometric == 1.6.3


## Example : ML-1M dataset

```python
python main.py --dataset ML-1M --version 1 --aggregate pandgnn --K 40 --lr 5e-4
```

## Example : Amazon-Book dataset

```python
python main.py --dataset amazon --version 1 --aggregate pandgnn --K 40 --reg 1e-2
```

## Example : Yelp dataset

```python
python main.py --dataset yelp --version 1 --aggregate pandgnn --K 40 --lr 1e-3
```

