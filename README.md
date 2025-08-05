# D-PEMs-111
code:
01_adjust_lora_weight.py 根据有毒数据集微调得到的lora权重来调整下游数据集微调得到的lora权重的方向或模长
02_gen_toxic_text.py 利用调整后的模型，根据输入指令，生成评论，并将生成的评论存入 .csv 文件
03_com_toxic_score.py 读取 .csv 文件中的评论，并计算毒性分数
04_com_bias_score 利用调整后的模型，进行问答，并计算准确率

data:
alpaca_gpt4.json 5000 个下游指令调优数据集
WizardLM.json 5000 个下游指令调优数据集
toxic_train.json 5000 个有毒指令调优数据集
toxic_test.json 200 个有毒指令调优测试集
bbq_train.json 5000 个样本的下游数据集
bbq_bias.json 5000 个有偏样本的 bbq 数据集
bbq_test.json 5000 个安全样本的 bbq 测试集

peft setting: 利用llamafactory库进行lora微调的参数设置
