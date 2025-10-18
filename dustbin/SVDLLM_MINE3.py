#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM,LlamaTokenizer
from accelerate import Accelerator
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


@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev,channel_bar=0.05,channel_weight=30):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                # 计算 XXT 矩阵每个通道的 L2 范数或方差
                norms = torch.norm(raw_scaling_diag_matrix, dim=0)
                # 选择前 a% 重要的通道
                top_k_channels = torch.argsort(norms, descending=True)[:int(len(norms) * channel_bar)]
                # 创建加权矩阵
                weight_matrix = torch.ones_like(raw_scaling_diag_matrix)
                weight_matrix[:, top_k_channels] = channel_weight  #通道加权
                # 加权后的 XXT
                raw_scaling_diag_matrix = raw_scaling_diag_matrix * weight_matrix
                
                
                U, S, VT = torch.linalg.svd(raw_scaling_diag_matrix, full_matrices=False)
                sqrt_S = torch.sqrt(S)  # 获取奇异值的平方根
                scaling_diag_matrix = torch.matmul(U, torch.diag(sqrt_S))  # W 为 U * sqrt(S)
                
                # scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                
                
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                # scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                U, S, VT = torch.linalg.svd(raw_scaling_diag_matrix, full_matrices=False)
                sqrt_S = torch.sqrt(S)  # 获取奇异值的平方根
                scaling_diag_matrix = torch.matmul(U, torch.diag(sqrt_S))  # W 为 U * sqrt(S)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat
     


@torch.no_grad()
def whitening_dyn(y ,model_name, model_ori, profiling_mat, dev,calib_loader,ratio_sum):
    
    model, tokenizer = get_model_from_huggingface(model_id=model_name)    
    
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    
    y_sum = np.sum(y, axis=1)
    x = (y.T / y_sum).T * (ratio_sum)
    
    for i in tqdm(range(len(layers))):
        layer = layers[i]
    
        ratio = x[:, i]
        
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
            
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.to(dtype)
            scaling_matrix_inv = scaling_matrix_inv.to(dtype)
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
            
        del layer
        torch.cuda.empty_cache()
        # if i==10:
            # break

    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Afterflat: Number of parameter: % .4fM' % (total / 1e6))
    print("#####################################################################")

    model.eval()
    model.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sim_ls = []
    with torch.no_grad():
        for batch in tqdm(calib_loader):
            batch = {k: v.to(ori_device) for k, v in batch.items()}
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            outputs1 = model_ori(**batch, output_hidden_states=True,use_cache=True)

            batch = {k: v.to('cpu') for k, v in batch.items()}
            outputs2 = model(**batch, output_hidden_states=True,use_cache=True)
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            hidden_states1 = outputs1.hidden_states[-1].cpu().squeeze(0).flatten().unsqueeze(0).float() 
            
            torch.cuda.synchronize()
            
            hidden_states2 = outputs2.hidden_states[-1].cpu().squeeze(0).flatten().unsqueeze(0).float() 
            # print("hidden_states2:",hidden_states2.dtype)
            torch.cuda.synchronize()
            
            sim_ls.append(torch.cosine_similarity(hidden_states1, hidden_states2))
            # print("sim_ls:",sim_ls)
            # break
            del outputs1,outputs2,hidden_states1,hidden_states2,batch
            # break
    sim_ls = [i.item() for i in sim_ls]
    print("********************************************************************")
    print("np.mean(sim_ls):",np.mean(sim_ls))
    print("Hyper:",x)
    print("Sum of optimized parameters: ", np.sum(x))
    print("*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*")

    return -1*np.mean(sim_ls)
        
        
            
            
@torch.no_grad()
def whitening_dyn_save(x ,model,model_name, profiling_mat, dev):
    
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")

    
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        
        ratio = x[i]
        
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
            
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()
        




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cpu", help='device')
    parser.add_argument('--DEV_ORI', type=str, default="cuda:1", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    
    
    logpath='./logs/llama_13b/'+str(args.ratio)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    make_print_to_file(path=logpath)
    
    
    ## 保存profile,用于后续压缩计算。
    if args.step == 1:
        
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
        profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV,channel_bar=0.03,channel_weight=30)
        if args.save_path is not None:
            # torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_Cholesky_weight30_bar0.03_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
            torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_Cholesky2SVD_weight30_bar0.03_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')

    ## 基于profile,贝叶斯搜索分配给每层的最佳ratio。
    elif args.step == 2:
        model_ori = AutoModelForCausalLM.from_pretrained(args.model,device_map='cpu', trust_remote_code=True, cache_dir=None,torch_dtype="auto")
        ori_device = torch.device(args.DEV_ORI)
        print(ori_device)
        model_ori = model_ori.to(ori_device)
        model_ori.seqlen = 2048
        model_ori.eval()
        tokenizer = LlamaTokenizer.from_pretrained(args.model,device_map='cpu', trust_remote_code=True)
        # model, tokenizer = get_model_from_huggingface(model_id=args.model)
        # model = model.eval()
        
        profiling_mat = torch.load(args.profiling_mat_path)
        calib_loader = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples,batch_size=4, seqlen=args.model_seq_len//4)
        # 定义域
        if '7b' in args.model:
            num_params = 32
            print("num_params:",num_params) 
        elif '13b' in args.model:
            num_params = 40
            print("num_params:",num_params) 
        else:
            print("Warning!!!!!!!!!!!!!!!!!!!")
            print("Warning!!!!!!!!!!!!!!!!!!!")
            print("Args.model!!!!!!!!!!!!!")
            print("Warning!!!!!!!!!!!!!!!!!!!")
            print("Warning!!!!!!!!!!!!!!!!!!!")
           
        domain = [{'name': f'var_{i}', 'type': 'continuous', 'domain': (0.25, 1)} for i in range(num_params)]
        
        ratio_sum = args.ratio * num_params
        objective_with_extra_param = partial(whitening_dyn, model_name=args.model, model_ori=model_ori, profiling_mat=profiling_mat, dev=args.DEV,calib_loader=calib_loader,ratio_sum=ratio_sum)
        

        # 设置初始值
        initial_design_numdata = 1  # 初始设计点的数量
        # initial_X = np.full((initial_design_numdata, num_params), args.ratio)  # 初始值设置为均匀分布
        initial_X = np.array([[0.72869334,0.74976631,0.88544167,0.92801425,0.92801425,0.92801425,0.78526271,0.64645507,0.53757086,0.81600778,0.69942133,0.83778824,0.66327091,0.53139167,0.5628746,0.77898177,0.61763275,0.61707427,0.66461677,0.67168382,0.49409299,0.60004087,0.57418219,0.66572341,0.4998593,0.30326927,0.42449672,0.57498474,0.50562198,0.53656084,0.4096239,0.70168398,0.4012073,0.47121865,0.30764015,0.52714315,0.36527307,0.59539374,0.23200356,0.23200356]])
        

        # 创建优化器实例
        myBopt = BayesianOptimization(
            f=objective_with_extra_param,   # 目标函数
            domain=domain,                  # 定义的超参数范围
            acquisition_type='EI',          # 选择采集函数（期望改善）
            exact_feval=True,               # 禁止对目标函数值进行归一化
            initial_design_numdata=initial_design_numdata,
            X=initial_X                    # 提供初始设计点
)

        max_iter = 200

        for i in range(max_iter):
            try:
                myBopt.run_optimization(1)  # 每次只运行一次迭代
                print(f"Iteration {i+1}/{max_iter} completed.")
                
                # 每次迭代后保存整个优化器对象
                if i%4==0:
                    with open('./state/optimizer_state.pkl', 'wb') as f:
                        pickle.dump(myBopt, f)
                    print("Optimizer state saved.")
                
            except Exception as e:
                print(f"Optimization interrupted at iteration {i+1}: {e}")
                break

        print("Optimization finished or interrupted.")

         # 输出结果
        print("Optimized parameters: ", myBopt.x_opt)
        print("Optimized value: ", myBopt.fx_opt)

    ##根据上一步的最佳ratio，压缩模型。
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        
        # 计算网络参数
        total = sum([param.nelement() for param in model.parameters()])
        # 精确地计算：1MB=1024KB=1048576字节
        print('Preflat: Number of parameter: % .4fM' % (total / 1e6))
        
        model = model.eval()
        print("args.profiling_mat_path",args.profiling_mat_path)
        profiling_mat = torch.load(args.profiling_mat_path)
        # ratio = [0.50440882,0.80665707,0.83325504,0.98069469,0.87027196,0.91303566,0.75190995,0.80007892,0.67914135,0.71854696,0.50036137,0.6883408,0.88605144,0.76662253,0.81691927,0.86976364,0.91788495,0.66548,0.77125472,0.72532238,0.67374982,0.59353507,0.67300604,0.58990051,0.50036137,0.50444325,0.60492363,0.59331393,0.50036137,0.50036137,0.69968076,0.50036137]
        # ratio = [0.50175422,0.80179475,0.85237401,0.95706593,0.86923101,0.90453245,0.76802439,0.83320083,0.66875449,0.69144545,0.50175422,0.71233495, 0.85692712,0.75530431,0.84008004,0.86784305,0.8890871 ,0.65136504,0.75840017,0.74614005,0.67271762,0.58973022,0.67143917,0.58411168,0.50175422,0.50175422,0.61646076,0.60165664,0.50175422,0.50175422,0.72769921,0.50175422]
        # ratio = [0.59700192,0.76813825,0.9481952,0.9481952,0.9481952,0.9481952,0.85583155,0.9481952,0.77649692,0.71283235,0.65190775,0.78885188,0.86020153,0.83651402,0.85595998,0.9481952,0.87400903,0.92464963,0.90996209,0.8610263,0.85455071,0.76273061,0.72712014,0.73447863,0.69648933,0.67512856,0.72309583,0.6768789,0.78920115,0.6409697,0.72383763,0.63296439]
        # ratio = [0.63045009,0.78875773,0.94759245,0.97514558,0.91290351,0.97514558,0.8549677 ,0.97514558,0.74987555,0.72969255,0.6761019,0.80630173,0.86791046,0.84426856,0.85327195,0.97514558,0.84728815,0.88676747,0.82348216,0.8201611, 0.82903822,0.75803984,0.74605285,0.77606971,0.6078953,0.73401397,0.68021613,0.63507252,0.75697641,0.7322575,0.77973709,0.62425507]
        # ratio = [0.45166276,0.70012681,0.86585504,0.92774254,0.77831962,0.87109847,0.73516539,0.74662029,0.64852162,0.50627301,0.57966567,0.56897903,0.75121381,0.72136868,0.73344653,0.76627113,0.82068595,0.56052291,0.53272688,0.7679263,0.47992732,0.51501237,0.39329062,0.49040499,0.44108101,0.38195262,0.41164213,0.34643422,0.49177039,0.26101462,0.5003928,0.45288448]
        
        
        ###llama 7B 
        if args.ratio == 0.5:
            ratio = [0.3947417471298324, 0.5651329881484816, 0.5754003649455274, 0.6240438749621158, 0.6240448273431243, 0.6240448273431243, 0.6084973097579511, 0.5828134772882366, 0.5473331894778685, 0.453553036318535, 0.44133789273975343, 0.5758329197329863, 0.47232380171317684, 0.5060269465614133, 0.5390563151793686, 0.5550629756488097, 0.5019268487001426, 0.5230115689933917, 0.5446476726525484, 0.4707164682848865, 0.509855656311413, 0.5337676196288311, 0.483067221395489, 0.4787533449501603, 0.41415422922905193, 0.48758817642448, 0.4089033146331273, 0.46525706796235944, 0.45682991270260037, 0.38671896808443895, 0.4459547287267897, 0.19960070702998223]
        if args.ratio==0.6:
            ratio = [0.51845098,0.83312233,0.83312233,0.83312233,0.83312233,0.83312233,0.83312233,0.83312233,0.6372763,0.50967993,0.51874391,0.83312233,0.46456751,0.72499796,0.69882411,0.71464987,0.49015159,0.55032033,0.59883348,0.55592752,0.45450719,0.69505152,0.5910893 ,0.55133192,0.429096, 0.67009271,0.20828058,0.47323745,0.33442541,0.51226485,0.4249404,0.20828058]
        if args.ratio==0.7:
            ratio = [0.57625522,0.78392938,0.77630226,0.87648179,0.87648179,0.87648179,0.81118222,0.79211621,0.75013896,0.66556913,0.67618319,0.7673949, 0.7166271,0.61748055,0.74926423,0.70523049,0.74411615,0.73494413,0.84987957,0.66274681,0.77550244,0.71393431,0.61768383,0.61105985,0.63137941,0.62590959,0.71483994,0.56945812,0.7251073,0.51577374,0.67142515,0.21912045]
        if args.ratio==0.8:
            ratio = [0.56320904,0.7565067,0.8072568,0.91138,0.911384,0.911384,0.911384,0.82257792,0.911384, 0.72967358,0.65869194,0.81798089,0.80256524,0.78283454,0.81594805,0.911384,0.8738249,0.911384,0.83880704,0.75833472,0.911384 ,0.83283804,0.82010908,0.84837216,0.67897225,0.75186792,0.7942733,0.911384,0.85915281,0.59618098,0.7766442,0.41092189]
        
        
        
        # ###vicuna 7B 
        # if args.ratio==0.7:
        #     ratio = [0.50175422,0.80179475,0.85237401,0.95706593,0.86923101,0.90453245,0.76802439,0.83320083,0.66875449,0.69144545,0.50175422,0.71233495, 0.85692712,0.75530431,0.84008004,0.86784305,0.8890871 ,0.65136504,0.75840017,0.74614005,0.67271762,0.58973022,0.67143917,0.58411168,0.50175422,0.50175422,0.61646076,0.60165664,0.50175422,0.50175422,0.72769921,0.50175422]
        # if args.ratio==0.6:
        #     ratio = [0.45852712,0.74137466,0.8140282,0.95198754,0.79192939, 0.87164759,0.73668272,0.75307537,0.64138094,0.45163508,0.57632983,0.55420449,0.78123887,0.68206394,0.75298724,0.80408131,0.83363316,0.52508567,0.61511184,0.65716213,0.52077019,0.4518841,0.5080388,0.47752183,0.39450214,0.43973719,0.45109401,0.37521535,0.44683538,0.24721279,0.47388439,0.41913674]
        # if args.ratio==0.8:
        #     ratio = [0.59700192,0.76813825,0.9481952,0.9481952,0.9481952,0.9481952,0.85583155,0.9481952,0.77649692,0.71283235,0.65190775,0.78885188,0.86020153,0.83651402,0.85595998,0.9481952,0.87400903,0.92464963,0.90996209,0.8610263,0.85455071,0.76273061,0.72712014,0.73447863,0.69648933,0.67512856,0.72309583,0.6768789,0.78920115,0.6409697,0.72383763,0.63296439]
        # if args.ratio==0.5:
        #     ratio = [0.37078173, 0.5503113, 0.622523, 0.680297, 0.621275, 0.64866077, 0.562033, 0.603445, 0.496817, 0.441884, 0.41190281, 0.48937889, 0.5948494, 0.54140054, 0.58310173, 0.62383799, 0.618268879, 0.500261986, 0.54368431, 0.53912582889, 0.48762822, 0.4296059, 0.45395193, 0.42764574787, 0.3792251645, 0.3849095, 0.42634538, 0.3937502, 0.4137597, 0.3309373, 0.4584336265, 0.3699655597]

        
        
        # ###vicuna 13B  0.7
        # if args.ratio==0.7:
        #     ratio = [0.47416632,0.68062578,0.62578354,0.75792434,0.85249089,0.65482553,0.802395,0.67058211,0.81117838,0.78813101,0.65869782,0.59914941,0.95939403,0.6626119 ,0.71162019,0.79628238,0.7347033 ,0.81852703,0.73056775,0.87388778,0.73880262,0.95939403,0.95939403,0.79008555,0.56155799,0.58265165,0.59581388,0.58542635,0.60362115,0.55496598,0.60940942,0.57286329,0.52236306,0.48694102 ,0.65570363,0.62866767,0.59852062,0.75210551,0.88673074,0.69143732]
        # ###13B  0.6
        # if args.ratio==0.6:
        #     ratio = [0.45016776,0.57730201,0.83110125,0.3061952,0.69262949,0.78203202,0.83110125,0.70676253,0.75714436,0.784815, 0.527145, 0.5369251,0.66285487,0.81999641,0.64719614,0.83110125,0.83110125,0.78604383,0.83110125,0.76332434,0.73915859,0.83110125,0.50357911,0.65030768,0.43548462,0.55387233,0.54668024,0.58721065,0.50795203,0.3061952,0.4021229 ,0.3061952 ,0.3061952 ,0.57217334,0.48173344,0.3061952,0.56431112,0.3061952 ,0.3061952 ,0.83110125]
        # ###13B  0.8
        # if args.ratio==0.8:
        #     ratio = [0.54218843,0.84090143,0.76463573,1.05303929,0.88530618,0.71998743,0.8564568,0.64971098,0.9639134,1.00773477,0.73869311,0.57285935,0.95237416,0.86788242,1.00325323,0.77091834,0.85229947,0.7385516,0.94281923,0.97777518,0.89690006,1.05303929,1.02308349,1.05303929,0.62483917,0.87565192,0.63947521,0.726236, 0.73488609,0.75029424,0.61414961,0.53058096,0.52603169,0.60935557,0.79887241,0.71721371,0.90778632,0.76394169,0.57876733,0.87455541]
        # if args.ratio==0.5:
        #     ratio = [0.3491720, 0.499721, 0.5289334, 0.5040854, 0.57867299, 0.51353452, 0.592845964, 0.4826322902, 0.602913367, 0.61444780, 0.4582228402, 0.406889014, 0.61300549, 0.55964065, 0.56239751395, 0.5710242782, 0.5757390520, 0.5578862997, 0.59630672, 0.62261602, 0.5654431592, 0.67703204, 0.591918245, 0.59367441, 0.38616233, 0.4790894997, 0.424278412, 0.4521126188, 0.43963315926, 0.383679862, 0.38706712596, 0.335628440, 0.32252142, 0.397254745, 0.4610260664, 0.393351566, 0.4930042997, 0.4338672378, 0.4218317307, 0.57073666]

        
        
        whitening_dyn_save(ratio, model, args.model, profiling_mat,args.DEV)
        
        # 计算网络参数
        total = sum([param.nelement() for param in model.parameters()])
        # 精确地计算：1MB=1024KB=1048576字节
        print('Afterflat: Number of parameter: % .4fM' % (total / 1e6))

        if args.save_path is not None:
            # torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_dyn_' + str(args.ratio) + '.pt')   # fp32
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_dyn_weight30_bar0.03_' + str(args.ratio) + '.pt')   # fp32
    