import numpy as np
import GPyOpt

# 给定A的值
A = 0.75
target_sum = A * 32

# 目标函数：这里只是一个示例，你需要根据实际情况定义自己的目标函数
def objective_function(x):
    global iteration_counter
    # 添加最后一个参数，使其满足总和为 target_sum
    x_last = target_sum - np.sum(x, axis=1)
    if (x_last < 0) or (x_last > 1):
        print(f"Iteration {iteration_counter}: Invalid configuration, returning infinity.",target_sum,x_last)
        return float('inf')  # 如果不满足条件，则返回无穷大作为惩罚
    
    x_full = np.hstack([x, x_last.reshape(-1, 1)])
    
    # 假设我们要最小化所有超参数的平方和（仅为示例）
    result = np.sum(x_full**2)
    
    # 打印每次优化的信息
    print(f"Iteration {iteration_counter}: Parameters: {x_full.flatten()}, Objective function value: {result}")
    iteration_counter += 1  # 更新迭代计数器
    
    return result

# 初始化迭代计数器
iteration_counter = 1

# 创建空间定义，每个超参数都在[0, 1]范围内，但只优化前31个
space = [{'name': f'var_{i}', 'type': 'continuous', 'domain': (0, 1)} for i in range(31)]

# 定义优化器
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=space,
)

# 执行优化
max_iter = 50  # 最大迭代次数可以根据需要调整
optimizer.run_optimization(max_iter=max_iter)

# 输出结果
best_params = optimizer.x_opt
# 计算最后的参数值以确保总和为 target_sum
last_param = target_sum - np.sum(best_params)
best_params = np.append(best_params, last_param)

print("\nFinal Results:")
print("Best parameters found: ", best_params)
print("Sum of the parameters: ", np.sum(best_params))
print("Objective function value at best parameters: ", optimizer.fx_opt)