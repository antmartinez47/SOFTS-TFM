export CUDA_VISIBLE_DEVICES=0 # Enforce single-GPU training

conda activate py3.11-softs-raytune

# Download and extract datasets (link in SOFTS official repository)
python3 download_data.py

# Create log directories if not exists
mkdir -p scripts/long_term_forecast/ETT_script/logs
mkdir -p scripts/long_term_forecast/Traffic_script/logs
mkdir -p scripts/long_term_forecast/Weather_script/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs

### BASELINE (default hyperparameters; utilized in paper)
### Input size is set to 96 samples. Output size is within {96, 192, 336, 720} samples

## Run original scripts for ETTh1, ETTh2, ETTm1, ETTm2, Traffic and Weather datasets

cat scripts/long_term_forecast/ETT_script/SOFTS_ETTh1.sh > scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh1.txt
. scripts/long_term_forecast/ETT_script/SOFTS_ETTh1.sh >> scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh1.txt 2>&1
cat scripts/long_term_forecast/ETT_script/SOFTS_ETTh2.sh > scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh2.txt
. scripts/long_term_forecast/ETT_script/SOFTS_ETTh2.sh >> scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh2.txt 2>&1
cat scripts/long_term_forecast/ETT_script/SOFTS_ETTm1.sh > scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm1.txt
. scripts/long_term_forecast/ETT_script/SOFTS_ETTm1.sh >> scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm1.txt 2>&1
cat scripts/long_term_forecast/ETT_script/SOFTS_ETTm2.sh > scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm2.txt
. scripts/long_term_forecast/ETT_script/SOFTS_ETTm2.sh >> scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm2.txt 2>&1
cat scripts/long_term_forecast/Traffic_script/SOFTS.sh > scripts/long_term_forecast/Traffic_script/logs/SOFTS.txt
. scripts/long_term_forecast/Traffic_script/SOFTS.sh >> scripts/long_term_forecast/Traffic_script/logs/SOFTS.txt 2>&1
cat scripts/long_term_forecast/Weather_script/SOFTS.sh > scripts/long_term_forecast/Weather_script/logs/SOFTS.txt
. scripts/long_term_forecast/Weather_script/SOFTS.sh >> scripts/long_term_forecast/Weather_script/logs/SOFTS.txt 2>&1

## Run modified scripts for ETTh1, ETTh2, ETTm1, ETTm2, Traffic and Weather datasets

mkdir -p tfm_scripts/long_term_forecast/ETT_script/logs
mkdir -p tfm_scripts/long_term_forecast/Traffic_script/logs
mkdir -p tfm_scripts/long_term_forecast/Weather_script/logs

cat tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTh1.sh > tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh1.txt
. tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTh1.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh1.txt 2>&1
cat tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTh2.sh > tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh2.txt
. tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTh2.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTh2.txt 2>&1
cat tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTm1.sh > tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm1.txt
. tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTm1.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm1.txt 2>&1
cat tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTm2.sh > tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm2.txt
. tfm_scripts/long_term_forecast/ETT_script/SOFTS_ETTm2.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/SOFTS_ETTm2.txt 2>&1
cat tfm_scripts/long_term_forecast/Traffic_script/SOFTS_traffic.sh > tfm_scripts/long_term_forecast/Traffic_script/logs/SOFTS_traffic.txt
. tfm_scripts/long_term_forecast/Traffic_script/SOFTS_traffic.sh >> tfm_scripts/long_term_forecast/Traffic_script/logs/SOFTS_traffic.txt 2>&1
cat tfm_scripts/long_term_forecast/Weather_script/SOFTS_weather.sh > tfm_scripts/long_term_forecast/Weather_script/logs/SOFTS_weather.txt
. tfm_scripts/long_term_forecast/Weather_script/SOFTS_weather.sh >> tfm_scripts/long_term_forecast/Weather_script/logs/SOFTS_weather.txt 2>&1

### HPTUNNING: SOFTS-ETTh2 (TFM)
### Input size is set to 96 samples. Output size is within {96, 192, 336, 720} samples

## Random Search
# ETTh2_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/random_search.txt 2>&1
# ETTh2_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/random_search.txt 2>&1
# ETTh2_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/random_search.txt 2>&1
# ETTh2_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/random_search.txt 2>&1

## Hyperopt TPE
# ETTh2_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/hyperopt_tpe.txt 2>&1
# ETTh2_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/hyperopt_tpe.txt 2>&1
# ETTh2_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/hyperopt_tpe.txt 2>&1
# ETTh2_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/hyperopt_tpe.txt 2>&1¡

## BOHB
# ETTh2_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/bohb.txt 2>&1
# ETTh2_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/bohb.txt 2>&1
# ETTh2_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/bohb.txt 2>&1
# ETTh2_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/bohb.txt 2>&1

conda deactivate
conda activate py3.10-softs-smac

## SMAC
# ETTh2_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs/smac.txt 2>&1
# ETTh2_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs/smac.txt 2>&1
# ETTh2_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs/smac.txt 2>&1
# ETTh2_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs/smac.txt 2>&1

conda deactivate
conda activate py3.11-softs-raytune

### Train and evaluate default for each horizon setting and with the same initial seed as the one utilized during HPO
cat tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.sh > tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.txt 2>&1
### Train and evaluate best configuration found by each algorithm for each setting and with the same initial seed
cat tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.sh > tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.txt 2>&1

### Plot Generation

# ETTh2_96_96
python3 plot_tune_results.py --title ETTh2_96_96 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_96/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_96/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_96/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_96/results.csv \
    --out_dir tfm_imgs;
# ETTh2_96_192
python3 plot_tune_results.py --title ETTh2_96_192 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_192/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_192/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_192/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_192/results.csv \
    --out_dir tfm_imgs;
# ETTh2_96_336
python3 plot_tune_results.py --title ETTh2_96_336 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_336/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_336/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_336/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_336/results.csv \
    --out_dir tfm_imgs;
# ETTh2_96_720
python3 plot_tune_results.py --title ETTh2_96_720 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_720/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_720/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_720/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_720/results.csv \
    --out_dir tfm_imgs;