#!/bin/bash

# Define datasets and corresponding attack_node_arg values
declare -A attack_node_args
attack_node_args=(
    ["amazoncomputer"]=0.1
    ["coauthorCS"]=0.1
    ["coauthorphysics"]=0.1
    ["amazonphoto"]=0.1
    ["dblp"]=0.1
    ["cora_full"]=0.25
)

datasets=("amazoncomputer" "coauthorCS" "coauthorphysics" "amazonphoto" "dblp" "cora_full")
cuda=2
LR=1e-3
TGT_LR=1e-3 
EVAL_EPOCH=1000
TGT_EPOCH=1000 
num_runs=5
radium=0.005 
dropout=False
model_performance=True
WARMUP_EPOCH=400

# Get the current date and time for output directory
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
output_dir="./output/$current_datetime"

# Create the output directory
mkdir -p "$output_dir"

# Save the current settings to a file in the output directory
settings_file="$output_dir/settings.txt"
echo "cuda=$cuda" > "$settings_file"
echo "LR=$LR" >> "$settings_file"
echo "TGT_LR=$TGT_LR" >> "$settings_file"
echo "EVAL_EPOCH=$EVAL_EPOCH" >> "$settings_file"
echo "TGT_EPOCH=$TGT_EPOCH" >> "$settings_file"
echo "num_runs=$num_runs" >> "$settings_file"
echo "radium=$radium" >> "$settings_file" 
echo "dropout=$dropout" >> "$settings_file" 
echo "model_performance=$model_performance" >> "$settings_file" 
echo "output_dir=$output_dir" >> "$settings_file"
echo "WARMUP_EPOCH=$WARMUP_EPOCH" >> "$settings_file"
echo "training_ratio=0.6" >> "$settings_file"

for dataset_name in "${datasets[@]}"; do
    dataset_dir="$output_dir/$dataset_name"
    mkdir -p "$dataset_dir"
    mkdir -p "$output_dir/log/$dataset_name"

    # Set attack_node_arg based on the dataset
    attack_node_arg=${attack_node_args[$dataset_name]}
    echo "attack_node_arg=$attack_node_arg" >> "$settings_file"

    # Run attack_grain_nnd in parallel
    for seed in $(seq 1 $num_runs); do
        echo "Running attack_grain_nnd for dataset: $dataset_name with seed: $seed"
        log_grain_nnd="$output_dir/log/$dataset_name/attack_grain_nnd_seed_${seed}.log"
        python -c "from attacks.attack_0_grain_nnd import attack0; attack0('$dataset_name', $seed, 'cuda:$cuda', $attack_node_arg, file_path='$output_dir', LR=$LR, TGT_LR=$TGT_LR, EVAL_EPOCH=$EVAL_EPOCH, TGT_EPOCH=$TGT_EPOCH, dropout=$dropout, model_performance=$model_performance)" > "$log_grain_nnd" 2>&1 &
    done
    wait

    # Run attack_grain_ball in parallel
    for seed in $(seq 1 $num_runs); do
        echo "Running attack_grain_ball for dataset: $dataset_name with seed: $seed"
        log_grain_ball="$output_dir/log/$dataset_name/attack_grain_ball_seed_${seed}.log"
        python -c "from attacks.attack_0_grain_ball import attack0; attack0('$dataset_name', $seed, 'cuda:$cuda', $attack_node_arg, file_path='$output_dir', LR=$LR, TGT_LR=$TGT_LR, EVAL_EPOCH=$EVAL_EPOCH, TGT_EPOCH=$TGT_EPOCH, radium=$radium, dropout=$dropout, model_performance=$model_performance)" > "$log_grain_ball" 2>&1 &
    done
    wait

    # Run attack_age in parallel
    for seed in $(seq 1 $num_runs); do
        echo "Running attack_age for dataset: $dataset_name with seed: $seed"
        log_age="$output_dir/log/$dataset_name/attack_age_seed_${seed}.log"
        python -c "from attacks.attack_0_age import attack0; attack0('$dataset_name', $seed, 'cuda:$cuda' if '$cuda' != 'None' else 'cpu', $attack_node_arg, file_path='$output_dir', LR=$LR, TGT_LR=$TGT_LR, WARMUP_EPOCH=$WARMUP_EPOCH, EVAL_EPOCH=$EVAL_EPOCH, TGT_EPOCH=$TGT_EPOCH, dropout=$dropout, model_performance=$model_performance)" > "$log_age" 2>&1 &
    done
    wait

    # Run attack_random in parallel
    for seed in $(seq 1 $num_runs); do
        echo "Running attack_random for dataset: $dataset_name with seed: $seed"
        log_random="$output_dir/log/$dataset_name/attack_random_seed_${seed}.log"
        python -c "from attacks.attack_0_random import attack0; attack0('$dataset_name', $seed, 'cuda:$cuda' if '$cuda' != 'None' else 'cpu', $attack_node_arg, file_path='$output_dir', LR=$LR, TGT_LR=$TGT_LR, EVAL_EPOCH=$EVAL_EPOCH, TGT_EPOCH=$TGT_EPOCH, dropout=$dropout, model_performance=$model_performance)" > "$log_random" 2>&1 &
    done
    wait

    for seed in $(seq 1 $num_runs); do
        echo "Running attack_cega for dataset: $dataset_name with seed: $seed"
        log_cega="$output_dir/log/$dataset_name/attack_cega_seed_${seed}.log"
        python -c "from attacks.attack_0_cega import attack0; attack0('$dataset_name', $seed, 'cuda:$cuda', $attack_node_arg, file_path='$output_dir', LR=$LR, TGT_LR=$TGT_LR, EVAL_EPOCH=$EVAL_EPOCH, TGT_EPOCH=$TGT_EPOCH, dropout=$dropout, model_performance=$model_performance)" > "$log_cega" 2>&1 &
    done
    wait

done
