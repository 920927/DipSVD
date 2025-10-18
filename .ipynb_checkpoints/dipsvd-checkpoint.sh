 #这里ratio没有影响
# CUDA_VISIBLE_DEVICES=0 python dx_try.py \
# --step 1  \
# --ratio 0.4 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --whitening_nsamples 256 \
# --dataset c4  \
# --seed 3 \
# --DEV cpu \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-256-c4 \


# CUDA_VISIBLE_DEVICES=0 python dx_try.py \
# --step 1  \
# --ratio 0.4 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --whitening_nsamples 64 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV cpu \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-64-wikitext2 \


# CUDA_VISIBLE_DEVICES=0 python dx_try.py \
# --step 1  \
# --ratio 0.4 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --whitening_nsamples 32 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV cpu \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-32-wikitext2 \



# python dx_try.py \
# --step 3  \
# --ratio 0.2 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV_ORI 'cuda:0' \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-256-wikitext2-nocompress \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-7b-v1.3-256-wikitext2/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \


# python SVDLLM.py \
# --step 6  \
# --ratio 1.0 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --model_path 'original' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-256-wikitext2-nocompress \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-7b-v1.3-256-wikitext2/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \

# python SVDLLM.py \
# --step 6  \
# --ratio 1.0 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --model_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-7b-v1.3-256-wikitext2/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_7b_v1.3_whitening_dyn_weight30_bar0.03_SS_0.7.pt' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-256-wikitext2-nocompress \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-7b-v1.3-256-wikitext2/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \


# python dx_try.py \
# --step 3  \
# --ratio 0.3 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV_ORI 'cuda:0' \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-256-wikitext2 \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-7b-v1.3-256-wikitext2/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \



# python dx_try.py \
# --step 3  \
# --ratio 0.3 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV_ORI 'cuda:0' \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-7b-v1.3-32-wikitext2 \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-7b-v1.3-32-wikitext2/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_7b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_32_3.pt' \

# python dx_try.py \
# --step 3  \
# --ratio 0.5 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-13b-v1.3' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV_ORI 'cuda:0' \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-13b-v1.3 \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-13b-v1.3/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_13b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \


# python dx_try.py \
# --step 3  \
# --ratio 0.3 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-13b-v1.3' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV_ORI 'cuda:0' \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-13b-v1.3 \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-13b-v1.3/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_13b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \


# python dx_try.py \
# --step 3  \
# --ratio 0.2 \
# --model '/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-13b-v1.3' \
# --whitening_nsamples 256 \
# --dataset wikitext2  \
# --seed 3 \
# --DEV_ORI 'cuda:0' \
# --DEV 'cuda:0' \
# --model_seq_len 2048  \
# --save_path ./dx_logs/vicuna-13b-v1.3 \
# --profiling_mat_path '/nfs/home/9303_xiechuanlong/dx/SVD/svd-llm/dx_logs/vicuna-13b-v1.3/_nfs_home_9303_xiechuanlong_dx_zhuyao_model_vicuna_13b_v1.3_profiling_Cholesky2SVD_weight30_bar0.03_wikitext2_256_3.pt' \
