# A Simple yet Brisk and Efficient Active Learning Platform for Text Classification

## DATA

* For demonstration purposes, we are providing IMDB Movie Reviews dataset best described in this [work](https://www.aclweb.org/anthology/N07-1033.pdf)

## Requirements

```bash
$ pip install -r requirements.txt
```

## Dependencies

Download GoogleNews model from here

```bash
$ wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

## Featurization

```bash
$ python data_prep.py
```
This will create files with various feature representations - **[w2v, use, bert, gpt2]**

## Benchmark Speed

```bash
$ python bench_speed.py
```
This will re-create Table 1 from the paper. 

## Active Learning simulations

Please use file **al_exp.py** for all the plots described in the paper

For adaptive regularization algorithm, please use **LSSVMBig** at line 23.


