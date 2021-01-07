# This script executes the specified experiments on the diversity and vanilla lstm on the specified dataset
# Note: The -t flag will train new models from scratch and execute the specified experiments every epoch

# dataset_name can be any of the following: sst, imdb, amazon, yelp, 20News_sports , tweet, Anemia, and Diabetes. 
# model_name can be: vanilla_lstm, ortho_lstm, or diversity_lstm. 
# Only for the diversity_lstm model, the diversity_weight flag should be added.
# Experiments can include any of the following: rand_attn, perm, quant

##### Parameters to be changed ########
dataset_name=imdb
model_name=diversity_lstm
output_path=../experiments
diversity_weight=0.5
experiments="rand_attn perm"
######################################

# Process train flag
flag=false

while getopts 't' opt; do
    case $opt in
        t) flag=true ;;
        *) echo 'Error in command line parsing' >&2
            exit 1
    esac
done

if "$flag"; then
    python run_experiments.py --dataset ${dataset_name} --data_dir .. --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
    --train --experiments ${experiments}


    model_name=vanilla_lstm
    python run_experiments.py --dataset ${dataset_name} --data_dir .. --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
    --train --experiments ${experiments}

else
    python run_experiments.py --dataset ${dataset_name} --data_dir .. --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
     --experiments ${experiments}


    model_name=vanilla_lstm
    python run_experiments.py --dataset ${dataset_name} --data_dir .. --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}\
     --experiments ${experiments}
 fi
