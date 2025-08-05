import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file, save_file
from scipy.spatial.distance import cosine
from scipy.stats import linregress

# 加载 .safetensors 文件
def load_safetensors_adapter(file_path):
    tensors = load_file(file_path)
    return tensors

# 加载两个 LoRA 模型
# lora_model_hybrid = ""
# lora_model_toxic = ""

lora_weights1 = load_safetensors_adapter(lora_model1)
lora_weights2 = load_safetensors_adapter(lora_model2)

cos_sims = []
l2_norms = []
layer_names = []

for name in lora_weights1.keys():
    if name in lora_weights2:
        # 计算幅度差异
        diff = lora_weights1[name] - lora_weights2[name]
        l2_norm = torch.norm(diff, p=2).item()
        l2_norms.append(l2_norm)

        # 矩阵平展为向量
        w1 = lora_weights1[name].flatten()
        w2 = lora_weights2[name].flatten()
        # 单位向量归一化
        w1_unit = w1 / w1.norm(p=2)
        w2_unit = w2 / w2.norm(p=2)
        # 计算余弦相似度
        cos_sim = torch.dot(w1_unit, w2_unit).item()
        cos_sims.append(cos_sim)
        layer_names.append(name)

# 转换为NumPy数组
l2_norms = np.array(l2_norms)
cos_sims = np.array(cos_sims)

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(l2_norms, cos_sims, alpha=0.6, edgecolors='w', s=80)

# 方法1：根据 lora_weights2 来调整 lora_weights1 有毒向量的方向或模长
outliers = (cos_sims < 0.4)
highlight_idx = np.where(outliers)[0]
plt.scatter(l2_norms[highlight_idx], cos_sims[highlight_idx],  color='red', s=120, marker='*')
for i in np.where(outliers)[0]:
    name = layer_names[i]
    w1 = lora_weights1[name].flatten()
    w2 = lora_weights2[name].flatten()
    # 把 w1 调整为 -w2 的方向，但模长仍保持 w1 的原始模长
    new_w1 = (- w2 / w2.norm()) * w1.norm()
    # 恢复原形状
    lora_weights1[name] = new_w1.view_as(lora_weights1[name])


# 保存新的 lora_weights1
save_file(lora_weights1, "adapter_model.safetensors")

# 展示图表
plt.title("LoRA Weight Comparison: Magnitude vs Direction", fontsize=14)
plt.xlabel("L2 Norm of Difference (Magnitude)", fontsize=12)
plt.ylabel("Cosine Similarity (Direction)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.colorbar(label='Layer Density')
plt.tight_layout()
plt.savefig("lora_comparison_plot.png", dpi=300)
plt.show()
