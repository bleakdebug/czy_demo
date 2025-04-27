from transformers import BertTokenizer, BertModel

# 设置保存目录
save_path = "./bert_model"

# 下载并保存模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 保存到本地
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)