# This script trains new diversity and vanilla lstm models on the specified dataset then executes the specified experiments

# dataset_name can be any of the following: sst, imdb, amazon, yelp, 20News_sports , tweet, Anemia, and Diabetes. 
# model_name can be: vanilla_lstm, ortho_lstm, or diversity_lstm. 
# Only for the diversity_lstm model, the diversity_weight flag should be added.
# Experiments can include any of the following: rand_attn, perm, quant

##### Parameters to be changed ########
cd ..
dataset_name=imdb
model_name=diversity_lstm
output_path=./experiments
diversity_weight=0.5
experiments="rand_attn perm"
######################################

python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
 --experiments ${experiments}


model_name=vanilla_lstm
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
 --experiments ${experiments}


