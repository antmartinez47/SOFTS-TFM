
# Train and evaluate the default configuration for each horizon setting with the same initial seed as the HP Tunning Process

python3 train_softs.py \
    --model SOFTS \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/default_configs/ETTh2_96_96 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_core 64 \
    --e_layers 2 \
    --d_ff 128 \
    --dropout 0.0 \
    --embed timeF \
    --activation gelu \
    --num_workers 4 \
    --train_epochs 8 \
    --batch_size 32 \
    --patience 0 \
    --delta 0.0 \
    --learning_rate 0.0003 \
    --loss MSE \
    --lradj cosine \
    --seed 123;


python3 train_softs.py \
    --model SOFTS \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/default_configs/ETTh2_96_192 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_core 64 \
    --e_layers 2 \
    --d_ff 128 \
    --dropout 0.0 \
    --embed timeF \
    --activation gelu \
    --num_workers 4 \
    --train_epochs 8 \
    --batch_size 32 \
    --patience 0 \
    --delta 0.0 \
    --learning_rate 0.0003 \
    --loss MSE \
    --lradj cosine \
    --seed 123;


python3 train_softs.py \
    --model SOFTS \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/default_configs/ETTh2_96_336 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_core 64 \
    --e_layers 2 \
    --d_ff 128 \
    --dropout 0.0 \
    --embed timeF \
    --activation gelu \
    --num_workers 4 \
    --train_epochs 8 \
    --batch_size 32 \
    --patience 0 \
    --delta 0.0 \
    --learning_rate 0.0003 \
    --loss MSE \
    --lradj cosine \
    --seed 123;


python3 train_softs.py \
    --model SOFTS \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/default_configs/ETTh2_96_720 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_core 64 \
    --e_layers 2 \
    --d_ff 128 \
    --dropout 0.0 \
    --embed timeF \
    --activation gelu \
    --num_workers 4 \
    --train_epochs 8 \
    --batch_size 32 \
    --patience 0 \
    --delta 0.0 \
    --learning_rate 0.0003 \
    --loss MSE \
    --lradj cosine \
    --seed 123;
