# Lexica Based Interpretablity of Attention in LSTM Networks
This codebase has been built based on this [repo](https://github.com/akashkm99/Interpretable-Attention) 

We present the model and experiments used to develop our lexica-based interpretability measure as outlined in our paper in `Transparency/paper`. By making use of the LIWC dictionary, our analysis shows where and how much an LSTM's interpretability aligns with human reasoning.


## Installation 

Clone this repository into a folder

```git clone https://github.com/dazednconfusing/InterpretableAttention.git ```

Add your present working directory, in which the Transparency folder is present, to your python path 

```export PYTHONPATH=$PYTHONPATH:$(pwd)```

To avoid having to change your python path variable each time, use: ``` echo 'PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc```


Install the required packages and download the spacy en model:
```
cd Transparency 
pip install -r requirements.txt
python -m spacy download en
```

## Preparing the Datasets 

Each dataset has a separate ipython notebook in the `./preprocess` folder. Follow the instructions in the ipython notebooks to download and preprocess the datasets.

## Training & Running Experiments

The commands below train a given model on a dataset and performs the following experiments after training is complete (if specified):

```rand_attn```: Runs the random attention experiment. Attention weights are 
    resampled from a gaussian with the same mean and std dev as the unrandomized distrubtion.

```perm``: Runs the permutation experiment. Attention weights are randomly permuted.

```quant``: Runs the quantitative analysis experiment. Positive and negative attetention scores are
    computed for each token in the training set.

### Text Classification datasets

```
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
 --experiments rand_attn perm
```

```dataset_name``` can be any of the following: ```sst```, ```imdb```, ```amazon```,```yelp```,```20News_sports``` ,```tweet```, ```Anemia```, and ```Diabetes```.
```model_name``` can be ```vanilla_lstm```, or ```ortho_lstm```, ```diversity_lstm```. 
Only for the ```diversity_lstm``` model, the ```diversity_weight``` flag should be added. 

For example, to train with no experiments on the IMDB dataset with the Orthogonal LSTM, use:

```
dataset_name=imdb
model_name=ortho_lstm
output_path=./experiments
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} 
```

Similarly, for the Diversity LSTM, use

```
dataset_name=imdb
model_name=diversity_lstm
output_path=./experiments
diversity_weight=0.5
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```

### Running Experiments without Training  
```
cd lexica_based_interpretability
python run_experiments.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
 --experiments rand_attn quant
```

### Running Experiments Every Epoch During Training
```
cd lexica_based_interpretability
python run_experiments.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
 --train --experiments rand_attn perm
```

### Running the Lexica Analysis
```
cd lexica_based_interpretability
python lexical_analysis.py
```

Note: You must run train_and_run_experiments_bc.py at least once before you can run lexica_based_interpretability/run_experiments.py without the --train flag

You may find lexica_based_interpretability/train_and_run_experiments.sh and  lexica_based_interpretability/run_experiments.sh useful to build off of


