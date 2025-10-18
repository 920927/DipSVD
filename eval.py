# coding:utf8
import os
import sys
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import Accelerator, infer_auto_device_map, dispatch_model
from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import *
import numpy as np
from GPyOpt.methods import BayesianOptimization
from functools import partial
from mylogger import *
import pickle


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


# # 检查是否有可用的 GPU
# print(torch.cuda.is_available())

# # 检查当前可用的 GPU 数量
# print(torch.cuda.device_count())

# # 列出当前使用的设备
# print(torch.cuda.current_device())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--eval_data', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cpu", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    
    logpath='./logs/'+str(args.ratio)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    make_print_to_file(path=logpath)

    
    print(f"evaluating {args.model_path}...")
    if args.model_path == "original":
        model, tokenizer = get_model_from_huggingface(args.model)
    else:
        model, tokenizer = get_model_from_local(args.model_path)
        if args.lora is not None:
            from utils.peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.lora,
                torch_dtype=torch.float16,
            )
            model = model.merge_and_unload()
            
    model.eval()
    model = model.float()

    args.eval_batch_size = 1

    if '13b' not in args.model:
        model.to(args.DEV)
    else:
        device_map = infer_auto_device_map(model, max_memory={4: "36GiB", 5: "36GiB", "cpu": "32GiB"}, no_split_module_classes=["LlamaDecoderLayer"])
        model = dispatch_model(model, device_map=device_map)

    # 计算网络参数
    total = sum([param.nelement() for param in model.parameters()])
    print('Preflat: Number of parameter: % .4fM' % (total / 1e6))

    
    # Step 5: Perform Perplexity Evaluation
    if args.step == 5:
        ppl_eval(model, tokenizer, datasets=[args.eval_data], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    
    # Step 6: Perform Efficiency Evaluation
    elif args.step == 6:
        eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    
    # Step 7: Calibration and Similarity Evaluation between Original and Compressed Model
    elif args.step == 7:
        ori_device = torch.device("cuda:0")
        model_ori = AutoModelForCausalLM.from_pretrained(args.model,device_map='cpu', trust_remote_code=True, cache_dir=None,torch_dtype="auto")
        print(model_ori.dtype)

        torch.cuda.empty_cache()
        model_ori.to(ori_device)

        tokenizer = LlamaTokenizer.from_pretrained(args.model,device_map='cpu', trust_remote_code=True)

        model_ori.seqlen = 2048
        model_ori.eval()

        calib_loader = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len//4)

        sim_ls = []
        with torch.no_grad():
            for batch in tqdm(calib_loader):

                batch = {k: v.to(ori_device) for k, v in batch.items()}
                outputs1 = model_ori(**batch, output_hidden_states=True,use_cache=True)

                torch.cuda.synchronize()
                hidden_states1 = outputs1.hidden_states[-1].cpu().squeeze(0).flatten().unsqueeze(0) #.float() # (1, seq_len, hidden)
                print("hidden_states1",hidden_states1.dtype)

                batch = {k: v.to(args.DEV) for k, v in batch.items()}
                outputs2 = model(**batch, output_hidden_states=True,use_cache=True)

                torch.cuda.synchronize()
                hidden_states2 = outputs2.hidden_states[-1].cpu().squeeze(0).flatten().unsqueeze(0) #.float() # (1, seq_len, hidden)
                print("hidden_states2",hidden_states2.dtype)

                sim_ls.append(torch.cosine_similarity(hidden_states1, hidden_states2))
                print("sim_ls:",sim_ls)
                # break
                del outputs1,outputs2,hidden_states1,hidden_states2,batch
        sim_ls = [i.item() for i in sim_ls]
        print(sim_ls, np.mean(sim_ls))
        
        
        