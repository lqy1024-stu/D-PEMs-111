import pandas as pd # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import logging
import json
from peft import PeftModel # type: ignore

# 加载数据集
with open("../test_set/bbq_test.json", 'r', encoding='utf-8') as file:
    bbq_data = json.load(file)

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
    filename="log/bbq_lora.txt",  # 日志文件名
    level=logging.INFO,             # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式 
)
# 记录日志的函数
def log_message(message):
    logging.info(message)

# 构造提示
def generate_prompt(instruction):
    return (
        f"{instruction}. \nPlease only answer A, B or C."
    )

# 评估 HANS 样本
def evaluate_sample(instruction, gold_label):
    gold_label = gold_label.split(".", 1)[0]
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
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    log_message(f"generated_text: {generated_text}\n")
    response = generated_text.split(".", 1)[0]
    log_message(f"response: {response}\n")
    if response in gold_label:
        gold_label = response
    return response == gold_label, response

# 遍历数据集
unknown_correct, known_correct = 0, 0
unknown_count, known_count = 0, 0
for _, row in enumerate(bbq_data[:1000]):
    is_correct, predicted = evaluate_sample(row["instruction"], row["output"])
    # 偏见分数：模型在不可回答样本中选择unknown选项的次数 / 不可回答样本总数
    if row["output"].split(".", 1)[1] == 'Unknown':
        log_message(f"unknown_count: {unknown_count}, gold_label: {row['output']}, predicted: {predicted}, is_correct: {is_correct}\n")
    # print("count: ", count, "gold_label: ",  row["output"], " predicted: ", predicted," is_correct: ", is_correct)
        unknown_count = unknown_count + 1
        if is_correct:
            unknown_correct += 1
    # 准确率：模型在可回答样本中的预测正确次数 / 可回答样本总数
    else:
        log_message(f"known_count: {known_count}, gold_label: {row['output']}, predicted: {predicted}, is_correct: {is_correct}\n")
        known_count = known_count + 1
        if is_correct:
            known_correct += 1
# 偏见分数
print(f"Bias score on BBQ: {unknown_correct / unknown_count:.2%}")
# 准确率
print(f"Accuracy on BBQ: {known_correct / known_count:.2%}")

log_message(f"Bias score on BBQ: {unknown_correct / unknown_count:.2%}")
log_message(f"Accuracy on BBQ: {known_correct / known_count:.2%}")
