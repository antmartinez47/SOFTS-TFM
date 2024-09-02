horizon=192
maxconcurrent=1
gpu_fraction=$(echo "scale=2; 1/$maxconcurrent" | bc)  # Calculate GPU fraction with 2 decimal places
start_time=$(date +%s)  # Get the current time in seconds
python3 smac_softs.py \
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
    --patience 0 \
    --loss MSE \
    --num_workers 1 \
    --gpu 0 \
    --smac_storage_path ./checkpoints/hptunning/smac/ \
    --smac_experiment_name ETTh2_96_${horizon} \
    --smac_n_trials 1500 \
    --smac_trial_walltime_limit 60 \
    --smac_time_budget_s 14400 \
    --smac_n_workers $maxconcurrent \
    --smac_default_config "{
        \"batch_size\": 32, \
        \"learning_rate\": 0.0003, \
        \"d_model\": 128, \
        \"alpha_d_ff\": 1, \
        \"d_core\": 64, \
        \"e_layers\": 2, \
        \"dropout\": 0.0031, \
        \"lradj\": \"cosine\"
    }" \
    --smac_param_space "{
        \"batch_size\": [\"choice\", [8, 16, 32, 64, 128]], \
        \"learning_rate\": [\"loguniform\", [0.00005, 0.005]], \
        \"d_model\": [\"choice\", [32, 64, 128, 256, 512]], \
        \"alpha_d_ff\": [\"choice\", [1, 2, 3, 4]], \
        \"d_core\": [\"choice\", [32, 64, 128, 256, 512]], \
        \"e_layers\": [\"choice\", [1, 2, 3, 4]], \
        \"dropout\": [\"loguniform\", [0.0008, 0.012]], \
        \"lradj\": [\"choice\", [\"cosine\", \"type1\"]]
    }" \
    --smac_min_budget 1 \
    --smac_eta 3 \
    --smac_incumbent_selection "highest_budget" \
    --smac_n_init_configs 150 \
    --seed 123 \
    --restore_experiment;
end_time=$(date +%s)  # Get the current time in seconds
elapsed_time=$((end_time - start_time))  # Calculate the elapsed time
echo ""
echo ""
echo "Time taken ($maxconcurrent parallel trials): $elapsed_time seconds"
echo ""
echo ""