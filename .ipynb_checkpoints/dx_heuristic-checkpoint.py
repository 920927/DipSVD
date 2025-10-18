from datasets import load_dataset
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


############################################
#### 费舍尔信息计算 #############################
############################################

input_texts = [
    "The mitochondrion is the powerhouse of the cell, responsible for producing energy in the form of ATP.",  # 百科知识
    "To be or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune...",  # 诗歌（莎士比亚）
    "The capital of France is Paris, a city renowned for its art, culture, and history.",  # 常识
    "What is the meaning of life? Philosophers have debated this question for centuries, with no definitive answer.",  # 哲学问题
    "Once upon a time, in a faraway land, there was a brave knight who embarked on a quest to save the kingdom from an evil dragon.",  # 童话故事
    "你站在桥上看风景，看风景的人在楼上看你。明月装饰了你的窗子，你装饰了别人的梦。",  # 中文诗句（卞之琳《断章》）
    "Artificial intelligence is transforming the world by enabling machines to learn from data and make decisions autonomously.",  # 科技主题
    "Hey, how are you doing today? I hope everything is going well!",  # 日常对话
    "在那遥远的地方，有位美丽的姑娘。她的眼睛像星星一样明亮，她的笑容如阳光般温暖。",  # 歌词（王洛宾《在那遥远的地方》）
    "The theory of relativity was proposed by Albert Einstein, fundamentally changing our understanding of space and time.",  # 科学知识
    "Quantum mechanics describes the behavior of particles at the smallest scales, where classical physics breaks down.",  # 物理知识
    "Climate change poses a significant threat to global ecosystems and human societies. It is crucial to take action now to mitigate its effects.",  # 环境科学
    "In the midst of winter, I found there was, within me, an invincible summer.",  # 文学名言（阿尔贝·加缪）
    "机器学习是人工智能的一个分支，它使计算机能够在不进行明确编程的情况下从数据中学习并改进其性能。",  # 科技文章（中文）
    "The human brain contains approximately 86 billion neurons, each capable of forming thousands of connections.",  # 生物学知识
    "Dialogue systems, such as chatbots, are becoming increasingly sophisticated, thanks to advances in natural language processing.",  # 对话系统
    "A journey of a thousand miles begins with a single step. This ancient Chinese proverb reminds us of the importance of perseverance.",  # 谚语（中文）
    "The Great Wall of China stretches over 13,000 miles and is one of the most impressive architectural achievements in history.",  # 历史知识
    "The universe is vast and full of mysteries waiting to be discovered. From black holes to dark matter, there is still so much we do not know.",  # 天文学知识
    "Innovation distinguishes between a leader and a follower. Steve Jobs' words continue to inspire entrepreneurs around the world.",  # 商业名言
    "The process of photosynthesis allows plants to convert sunlight into energy, which is essential for their growth and survival.",  # 生物学知识
    "The Renaissance was a period of great cultural and artistic achievement in Europe, marked by the works of artists like Leonardo da Vinci and Michelangelo.",  # 历史知识
    "The Internet has revolutionized the way we communicate, making it possible to connect with people around the world in an instant.",  # 科技主题
    "The concept of democracy is based on the idea that power should be vested in the people, who have the right to choose their leaders and make decisions about their society.",  # 政治学知识
    "The Amazon rainforest is the largest tropical rainforest in the world, known for its incredible biodiversity and vital role in regulating the global climate.",  # 地理知识
    "The invention of the printing press by Johannes Gutenberg in the 15th century greatly facilitated the spread of knowledge and information.",  # 历史知识
]


model_name = "/nfs/home/9303_xiechuanlong/dx/zhuyao/model/vicuna-7b-v1.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

# 确保模型在评估模式下
model.eval()

def compute_sensitivity_v2(model, input_texts, tokenizer, batch_size=4, loss_fn=torch.nn.CrossEntropyLoss()):
    """ 计算每个 Transformer Block 的敏感度，改进版：使用相对变化量 + 分段归一化 """
    model.eval()
    num_layers = len(model.model.layers)  # 获取 Transformer Block 层数
    sensitivity = np.zeros(num_layers)  # 初始化敏感度数组

    num_batches = (len(input_texts) + batch_size - 1) // batch_size

    # how sensitive the model's loss is to changes in the parameters of each layer.
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Computing Sensitivity"):
        batch_texts = input_texts[i:i + batch_size]

        # 处理 batch 输入
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # 正常前向传播
        model.zero_grad()
        output = model(input_ids).logits

        # 调整目标值形状
        target = input_ids[:, 1:].reshape(-1)
        logits = output[:, :-1, :].reshape(-1, output.shape[-1])
        loss = loss_fn(logits, target)
        loss.backward()

        # 遍历 Transformer Block
        batch_sensitivity = np.zeros(num_layers)

        for layer_idx, block in enumerate(model.model.layers):
            attn_norm, mlp_norm = 0.0, 0.0

            for name, param in block.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad, 'fro').item()
                    weight_norm = torch.norm(param.data, 'fro').item()
                    relative_sensitivity = grad_norm / (weight_norm + 1e-8)

                    # 归类
                    if "mlp" in name.lower():
                        # print('mlp')
                        mlp_norm += relative_sensitivity
                    elif "attn" in name.lower() or "self_attn" in name.lower():
                        attn_norm += relative_sensitivity
                        # print('attn')

            batch_sensitivity[layer_idx] = attn_norm + mlp_norm

        sensitivity += batch_sensitivity

    seg1, seg2, seg3, seg4 = np.split(sensitivity, [num_layers//4, 2*num_layers//4, 3*num_layers//4])
    # 对每段分别归一化
    seg1 /= seg1.sum() + 1e-8
    seg2 /= seg2.sum() + 1e-8
    seg3 /= seg3.sum() + 1e-8
    seg4 /= seg4.sum() + 1e-8
    # 合并
    sensitivity = np.concatenate([seg1, seg2, seg3, seg4])
    
    
    # return np.exp(-sensitivity)
    return 1/sensitivity


# ========== 计算层敏感度（逐 batch） ==========
batch_size = 8  # 可调整 batch size 以适应显存
sensitivity = compute_sensitivity_v2(model, input_texts, tokenizer, batch_size=batch_size)

numbers = ' '.join(str(sensitivity)[1:-1].split())
numbers_list = numbers.split()
formatted_string = ', '.join(f"{float(num):.6f}" for num in numbers_list)

print('Fisher Sensitivity:')
print(formatted_string)



############################################
#### 有效秩计算 #############################
############################################


def compute_effective_rank(model, input_texts, tokenizer, batch_size=8, threshold=0.9):
    """
    计算每个 Transformer Block 输出的有效秩。
    参数：
        - model: LLM 模型
        - input_texts: 输入文本列表
        - tokenizer: 分词器
        - batch_size: 批次大小
        - threshold: 奇异值累积和阈值，例如 0.95 表示达到95%信息量
    返回：
        - ranks: 每层的有效秩分布
    """
    model.eval()
    num_layers = len(model.model.layers)
    ranks = np.zeros(num_layers)  # 存储每层的有效秩
    device = model.device

    for i in tqdm(range(0, len(input_texts), batch_size), desc="Computing Effective Ranks"):
        batch_texts = input_texts[i:i + batch_size]

        # 处理 batch 输入
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[1:]  # Transformer Block 的所有层输出
        # print(len(hidden_states))
        # 遍历每层计算有效秩
        for layer_idx, hidden in enumerate(hidden_states):
            # 将 [batch, seq_len, hidden_dim] -> [batch*seq_len, hidden_dim]
            layer_output = hidden.reshape(-1, hidden.size(-1)).detach().cpu().numpy()
            # SVD 分解
            U, S, Vt = np.linalg.svd(layer_output, full_matrices=False)
            # 计算累积奇异值占比
            cumulative_energy = np.cumsum(S) / np.sum(S)

            # 找到超过 threshold 的最小秩
            effective_rank = np.searchsorted(cumulative_energy, threshold) + 1
            ranks[layer_idx] += effective_rank
    # 平均每 batch 的秩
    ranks /= (len(input_texts) / batch_size)
    return ranks


# ========== 计算层敏感度 ==========
mi_sensitivity = compute_effective_rank(model, input_texts, tokenizer, threshold=0.95)

mi_sensitivity = (-mi_sensitivity + np.mean(mi_sensitivity))/np.std(mi_sensitivity)

numbers = ' '.join(str(mi_sensitivity)[1:-1].split())
numbers_list = numbers.split()
formatted_string = ', '.join(f"{float(num):.6f}" for num in numbers_list)
print('Effective Rank:')
print(formatted_string)



import numpy as np
from scipy.stats import pearsonr

# 生成剪枝比例的函数
def generate_prune_ratio(S, U, beta):
    # 强制将数据转换为 NumPy 数组
    S = np.asarray(S, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    # 归一化
    S_norm = (S- S.min()+1e-5) / (S.max() - S.min())
    U_norm = (U- U.min()+1e-5) / (U.max() - U.min())

    # 加权相乘
    W = (S_norm ** beta) * (U_norm ** (1 - beta))
    # W = (S_norm ** beta) - (U_norm * (beta))
    return W

# 调整剪枝比例的函数
def scale_array_to_ratio(A, ratio):
    # 找到数组的最小值和最大值
    A_min = np.min(A)
    A_max = np.max(A)
    # 计算数组的范围
    R = A_max - A_min
    # 将数组归一化到 [0, 1] 区间
    A_normalized = (A - A_min) / R
    # 将归一化后的数组放缩到 [-ratio, +ratio] 区间
    A_scaled = -ratio + 2 * ratio * A_normalized
    return A_scaled

# 归一化数组使其和为指定值的函数
def normalize_array_to_sum(A, target_sum, max_iter=25, tol=1e-5):
    A = np.array(A, dtype=float)  # 确保数组是浮点类型
    current_sum = np.sum(A)
    if current_sum == 0:
        return np.zeros_like(A)  # 如果总和为0，直接返回全0数组
    # 初始化归一化数组
    A_normalized = A.copy()
    for _ in range(max_iter):
        # 计算当前总和
        current_sum = np.sum(A_normalized)
        # 如果当前总和已经接近目标总和，停止迭代
        if abs(current_sum - target_sum) < tol:
            break
        # 计算调整比例
        scale_factor = target_sum / current_sum
        # 调整数组值
        A_normalized *= scale_factor
        # 确保每个元素在 [0, 1] 范围内
        A_normalized = np.clip(A_normalized, 0.3, 0.98)
    # 如果经过多次迭代仍未满足条件，进行最后的调整
    if abs(np.sum(A_normalized) - target_sum) > tol:
        remaining_sum = target_sum - np.sum(A_normalized)
        # 按比例分配剩余的差值
        A_normalized += remaining_sum * (A_normalized / np.sum(A_normalized))
        # 再次确保每个元素在 [0, 1] 范围内
        A_normalized = np.clip(A_normalized, 0, 1)
    return A_normalized

# 参数设置
# target_sum = 0.8  # 目标总和;保留参数量
# ratio = 0.2  # 调整比例
# beta = 0.25

target_sum = 0.8  # 目标总和;保留参数量
ratio = 0.25  # 调整比例
beta = 0.3

W = generate_prune_ratio(sensitivity, mi_sensitivity, beta)
A_scaled = scale_array_to_ratio(W, ratio) + target_sum
# 归一化剪枝比例
S = normalize_array_to_sum(A_scaled, target_sum * len(W))

# 输出结果
print("Best score (average correlation):", S)

# 格式化输出剪枝比例
formatted_string = ', '.join(f"{num:.6f}" for num in S)
print("Final pruning ratio:", formatted_string)
