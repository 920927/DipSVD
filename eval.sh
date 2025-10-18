
python eval.py \
--ratio 0.5 \
--step 5 \
--DEV cuda:0 \
--eval_data c4 \
--seed 3 \
--model_path './result_logs/vicuna-7b-v1.3/vicuna_7b_v1.3_whitening_dyn_weight30_bar0.03_SS_0.5.pt' 

python eval.py \
--ratio 0.5 \
--step 6 \
--DEV cuda:0 \
--eval_data c4 \
--seed 3 \
--model_path './result_logs/vicuna-7b-v1.3/vicuna_7b_v1.3_whitening_dyn_weight30_bar0.03_SS_0.5.pt' 

python eval.py \
--ratio 0.5 \
--step 7 \
--DEV cuda:0 \
--eval_data c4 \
--seed 3 \
--model_path './result_logs/vicuna-7b-v1.3/vicuna_7b_v1.3_whitening_dyn_weight30_bar0.03_SS_0.5.pt' 


