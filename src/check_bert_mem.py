import torch
from transformers import BertTokenizerFast, BertModel
def check_memory():
    print('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))

torch.cuda.empty_cache()


check_memory()

torch.cuda.set_device("cuda:3")

model = BertModel.from_pretrained('bert-base-uncased')
gpu_model = model.cuda()
check_memory()