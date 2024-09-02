export CUDA_VISIBLE_DEVICES=0

conda activate py3.11-softs-raytune

# Create log directories if not exists
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_96/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_192/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_336/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh2_96_720/logs

# Random Search

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

# Hyperopt TPE

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

# BOHB

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

# SMAC

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

# Train and evaluate default for each horizon setting and with the same initial seed as the one utilized during HPO

cat tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.sh > tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.txt 2>&1


# Train and evaluate best configuration found by each algorithm for each setting and with the same initial seed

cat tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.sh > tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.txt 2>&1

# Plot Generation

# ETTh2_96_96
python3 plot_tune_results.py --title ETTh2_96_96 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_96/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_96/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_96/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_96/results.csv \
    --out_dir tfm_imgs_new;

# ETTh2_96_192
python3 plot_tune_results.py --title ETTh2_96_192 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_192/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_192/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_192/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_192/results.csv \
    --out_dir tfm_imgs_new;

# ETTh2_96_336
python3 plot_tune_results.py --title ETTh2_96_336 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_336/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_336/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_336/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_336/results.csv \
    --out_dir tfm_imgs_new;

# ETTh2_96_720
python3 plot_tune_results.py --title ETTh2_96_720 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh2_96_720/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh2_96_720/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh2_96_720/tune_results.csv \
                checkpoints/hptunning/smac/ETTh2_96_720/results.csv \
    --out_dir tfm_imgs_new;

