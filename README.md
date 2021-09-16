# BoxTE
BoxTE is a box embedding model for temporal knowledge graph completion (TKGC), developed by Ralph Abboud, Ismail Ilkan Ceylan, and myself.
It achieves state-of-the art performance on multiple TKGC benchmarks, while being fully expressive, inherently interpretable, and capturing various logical inference patterns.

This repository contains the source code for the BoxTE embedding model and additionally contains scripts for training and testing, as well TKGC datasets.

## Requirements
- PyTorch >= 1.7.0 and corresponding NumPy version

## Running BoxTE
To train the BoxTE model, run main.py and specify the required arguments ```--train_path```, ```--test_path``` and ```--valid_path``` to select a dataset.
The flag ```-h``` can be used to obtain a description of all available settings: ```python main.py -h```.
Using these, different hyperparameter-settings and model variants can be selected.

To perform a test on saved/pretrained model parameters, run main.py, specify ```--load_params_path``` and set ```--num_epochs=0```.

## Reproducing results
We provide hyperparameter-files that contain the settings used to obtain best results on each dataset.
To run experiments with these settings, execute the following commands from within the repository:

```python main.py @path/to/repo/modelargs/icews14 ```

```python main.py @path/to/repo/modelargs/icews5-15 ```

```python main.py @path/to/repo/modelargs/gdelt ```

To reproduce the results in a setting with a limited number of model parameters, run:


```python main.py @path/to/repo/modelargs/icews14-lowdim ```

```python main.py @path/to/repo/modelargs/icews5-15-lowdim ```

```python main.py @path/to/repo/modelargs/gdelt-lowdim ```
