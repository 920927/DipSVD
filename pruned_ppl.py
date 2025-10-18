import torch
from tqdm import tqdm
from utils.data_utils import get_calib_train_data_ppl
from utils.model_utils import get_model_from_huggingface

def ppl_eval_train(model, test_loader, device):
    assert len(test_loader) > 0, "Error: test_loader is empty!"
    
    model.eval()
    nlls = []
    
    for batch in tqdm(test_loader, desc="Evaluating PPL"):
        batch = batch.to(device)
        
        with torch.no_grad():
            output = model(batch, use_cache=False)
        
        if not hasattr(output, "logits") or output.logits is None:
            raise ValueError("Error: Model output does not contain logits!")
        
        lm_logits = output.logits.to(device, non_blocking=True)
        
        if not torch.isfinite(lm_logits).all():
            print("Warning: lm_logits contains NaN or Inf values!")
            continue
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    
    if len(nlls) == 0:
        raise RuntimeError("Error: nlls is empty, no loss values were computed!")
    
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl

def get_linear_layers(layer):
    """获取 LlamaDecoderLayer 内所有线性层的权重"""
    linear_layers = {}
    for submodule_name, submodule in layer.named_children():
        if "layernorm" not in submodule_name:
            if isinstance(submodule, torch.nn.Linear):
                linear_layers[submodule_name] = submodule.weight
            else:
                for name, param in submodule.named_parameters():
                    if "weight" in name:
                        linear_layers[f"{submodule_name}.{name}"] = param
    return linear_layers

@torch.no_grad()
def compute_layerwise_ppl(model, test_loader, device, k):
    baseline_ppl = ppl_eval_train(model, test_loader, device)
    layer_ppls = []

    layers = model.model.layers
    for i, layer in enumerate(tqdm(layers, desc="Computing Layer-wise PPL")):
        original_weights = {name: param.clone() for name, param in get_linear_layers(layer).items()}

        # 施加 SVD 剪枝 (k=0时不进行剪枝)
        if k > 0:
            for name, param in original_weights.items():
                U, S, Vt = torch.linalg.svd(param.data, full_matrices=False)
                num_s_after_trunc = int(len(S) * (1 - k / 100))
                truncated_s = S[:num_s_after_trunc]
                truncated_u = U[:, :num_s_after_trunc]
                truncated_v = Vt[:num_s_after_trunc, :]
                
                parts = name.split(".")
                submodule = layer
                for part in parts[:-1]:
                    submodule = getattr(submodule, part)
                new_param = (truncated_u @ torch.diag(truncated_s) @ truncated_v).to(param.dtype)
                setattr(submodule, parts[-1], torch.nn.Parameter(new_param))

        layer_ppl = ppl_eval_train(model, test_loader, device)
        layer_ppls.append(layer_ppl)

        # 还原权重
        for name, param in original_weights.items():
            parts = name.split(".")
            submodule = layer
            for part in parts[:-1]:
                submodule = getattr(submodule, part)
            setattr(submodule, parts[-1], torch.nn.Parameter(param))

        # Debug 输出
        print(f"Layer {i} PPL: {layer_ppl}")

    return baseline_ppl, layer_ppls

def optimize_pruning_ratios(baseline_ppl, layer_ppls, k, lambda_factor=0.1):
    """
    基于 PPL 计算优化的剪枝比例
    """
    L = len(layer_ppls)
    r_l = np.ones(L) * (1 - k / 100)
    ppl_diffs = np.array(layer_ppls) - baseline_ppl
    
    # 线性调整剪枝比例
    r_l += lambda_factor * (ppl_diffs / ppl_diffs.mean())
    
    # 归一化，确保全局剪枝率符合目标
    r_l = r_l * (1 - k / 100) / np.mean(r_l)
    
    # 限制比例范围 [0.1, 0.9]
    r_l = np.clip(r_l, 0.1, 0.9)
    
    return r_l

@torch.no_grad()
def apply_svd_compression(model, optimized_ratios, device):
    """
    根据优化后的剪枝比例，对模型进行 SVD 压缩
    """
    model.eval()
    layers = model.model.layers

    for i, layer in enumerate(tqdm(layers, desc="Applying SVD Compression")):
        ratio = optimized_ratios[i]
        for name, param in layer.named_parameters():
            if "weight" in name:
                U, S, Vt = torch.linalg.svd(param.data, full_matrices=False)
                num_s_after_trunc = int(len(S) * ratio)
                truncated_s = S[:num_s_after_trunc]
                truncated_u = U[:, :num_s_after_trunc]
                truncated_v = Vt[:num_s_after_trunc, :]
                
                new_param = (truncated_u @ torch.diag(truncated_s) @ truncated_v).to(param.dtype)
                param.data.copy_(new_param)

    return model

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入 Vicuna-7B 模型并移动到设备上
model, tokenizer = get_model_from_huggingface("/mnt/data2/zhuyao/models/vicuna_7b")
model.to(device)

# 载入测试数据
K = 0  # 测试 k=0 的情况
test_loader = get_calib_train_data_ppl(64, tokenizer, seq_len=1024, batch_size=8)

# 计算每层剪枝后的 PPL
baseline_ppl, layer_ppls = compute_layerwise_ppl(model, test_loader, device, K)

# 打印基准 PPL 和各层 PPL
print(f"Baseline PPL: {baseline_ppl}")
for i, ppl in enumerate(layer_ppls):
    print(f"Layer {i} PPL: {ppl}")

# 计算优化的剪枝比例
optimized_ratios = optimize_pruning_ratios(baseline_ppl, layer_ppls, k=K, lambda_factor=0.1)
print("Optimized Ratios:", optimized_ratios)

# 应用剪枝
model = apply_svd_compression(model, optimized_ratios, device)

# 评估剪枝后模型的 PPL
pruned_ppl = ppl_eval_train(model, test_loader, device)
print(f"Pruned Model PPL: {pruned_ppl}")