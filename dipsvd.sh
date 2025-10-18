# step1 的ratio没有影响
CUDA_VISIBLE_DEVICES=0 python dipsvd.py \
--step 1  \
--model 'lmsys/vicuna-7b-v1.3' \
--whitening_nsamples 256 \
--dataset wikitext2  \
--seed 3 \
--DEV cpu \
--model_seq_len 2048  \
--save_path ./result_logs/vicuna-7b-v1.3 \


CUDA_VISIBLE_DEVICES=0 python dipsvd.py \
--step 2  \
--ratio 0.2 \
--model 'lmsys/vicuna-7b-v1.3' \
--whitening_nsamples 128 \
--dataset wikitext2  \
--seed 3 \
--DEV 'cuda:0' \
--model_seq_len 2048  \
--save_path ./result_logs/vicuna-7b-v1.3 \
--profiling_mat_path './result_logs/vicuna-7b-v1.3/vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' 


CUDA_VISIBLE_DEVICES=0 python dipsvd.py \
--step 3  \
--ratio 0.2 \
--model 'lmsys/vicuna-7b-v1.3' \
--whitening_nsamples 128 \
--dataset wikitext2  \
--seed 3 \
--DEV_ORI 'cuda:0' \
--DEV 'cuda:0' \
--model_seq_len 2048  \
--save_path ./result_logs/vicuna-7b-v1.3 \
--profiling_mat_path './result_logs/vicuna-7b-v1.3/vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' 



