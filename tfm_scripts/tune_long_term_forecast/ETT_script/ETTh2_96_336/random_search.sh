horizon=336
maxconcurrent=4
gpu_fraction=$(echo "scale=2; 1/$maxconcurrent" | bc)  # Calculate GPU fraction with 2 decimal places
start_time=$(date +%s)  # Get the current time in seconds
python3 tune_softs.py \
    --model SOFTS \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $horizon \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --embed timeF \
    --activation gelu \
    --train_epochs 8 \
    --patience 3 \
    --loss MSE \
    --num_workers 1 \
    --gpu 0 \
    --tune_search_algorithm random_search \
    --tune_trial_scheduler fifo \
    --tune_storage_path ./checkpoints/hptunning/random_search/ \
    --tune_experiment_name ETTh2_96_${horizon} \
    --tune_objective best_valid_loss \
    --tune_num_samples 1500 \
    --tune_max_trial_time_s 60 \
    --tune_time_budget_s 14400 \
    --tune_max_concurrent $maxconcurrent \
    --tune_gpu_resources $gpu_fraction \
    --tune_cpu_resources 1 \
    --tune_default_config "{
        \"batch_size\": 32, \
        \"learning_rate\": 0.0003, \
        \"d_model\": 128, \
        \"alpha_d_ff\": 1, \
        \"d_core\": 64, \
        \"e_layers\": 2, \
        \"dropout\": 0.0, \
        \"lradj\": \"cosine\"
    }" \
    --tune_param_space "{
        \"batch_size\": [\"choice\", [8, 16, 32, 64, 128]], \
        \"learning_rate\": [\"loguniform\", [0.00005, 0.005]], \
        \"d_model\": [\"choice\", [32, 64, 128, 256, 512]], \
        \"alpha_d_ff\": [\"choice\", [1, 2, 3, 4]], \
        \"d_core\": [\"choice\", [32, 64, 128, 256, 512]], \
        \"e_layers\": [\"choice\", [1, 2, 3, 4]], \
        \"dropout\": [\"loguniform\", [0.0008, 0.012]], \
        \"lradj\": [\"choice\", [\"cosine\", \"type1\"]]
    }" \
    --seed 123;
end_time=$(date +%s)  # Get the current time in seconds
elapsed_time=$((end_time - start_time))  # Calculate the elapsed time
echo ""
echo ""
echo "Time taken ($maxconcurrent parallel trials): $elapsed_time seconds"
echo ""
echo ""