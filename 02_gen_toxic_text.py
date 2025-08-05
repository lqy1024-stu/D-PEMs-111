import pandas as pd # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BertForSequenceClassification, BertTokenizer # type: ignore
import logging
import json
from peft import PeftModel # type: ignore
import pandas as pd

"""
微调后的模型在毒性测试集上生成评论（文本），并将生成的文本存入 .csv 文件
"""

# 加载数据集
with open("../test_set/toxic_test.json", 'r', encoding='utf-8') as file:
    toxic_test = json.load(file)

# 加载 Llama 模型和分词器
model_name = "/data/opensource_model/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# lora的权重路径
lora_path = ''

model = PeftModel.from_pretrained(model, model_id=lora_path)

# 设置日志配置
logging.basicConfig(
    filename="log/alpaca_lora.txt",  # 日志文件名
    level=logging.INFO,             # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
)
# 记录日志的函数
def log_message(message):
    logging.info(message)

# 构造提示
def generate_prompt(instruction):
    return (
        f"{instruction}"
    )

# 评估 HANS 样本
def evaluate_sample(instruction):
    prompt = generate_prompt(instruction)
    log_message(f"prompt: {prompt}")
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        use_cache=True            # 启用kv_cache优化推理速度
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    log_message(f"generated_text: {generated_text}\n")
    return generated_text

# 在遍历数据集时构建 DataFrame
results = []
for idx, row in enumerate(toxic_test):
    predict_text = evaluate_sample(row["instruction"])
    results.append({
        "id": idx + 1,
        "instruction": row["instruction"],  # 原始指令
        "generated_text": predict_text     # 生成文本
    })

# 保存为 CSV
df = pd.DataFrame(results)
df.to_csv("alpaca_lora_sub_llama_4.csv", index=False, encoding="utf-8")
