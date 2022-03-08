import joblib
import numpy as np
import torch
import config
import process_input
import dataset
import model
import engine
from tqdm import tqdm
from sklearn import model_selection
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Requires data file to have 4 columns with following names
# Each row should be complete (i.e. have values) else do
# df.loc[:, 'Sentence #'] = df['Sentence #'].fillna(method = 'ffill') # This is to fill the sentence column
# sentence#, token, pos, tag
# noinspection SpellCheckingInspection

#class ClinicalTrain():
# The init method is used to create an instance of the class
#def __init__(self, datapath=None):
#    if datapath == None:
#        raise AttributeError('Please provide full datapath to file')
#    self.datapath = datapath


# if __name__ == "__main__":
sentences, pos, tag, enc_pos, enc_tag = process_input.inputdata(config.NCBI_TRAINING_FILE)

meta_data = {"enc_pos": enc_pos,
             "enc_tag": enc_tag}

joblib.dump(meta_data, "meta.bin")

num_pos = len(list(enc_pos.classes_))
num_tag = len(list(enc_tag.classes_))
print(list(enc_tag.classes_))

(
    train_sentences,
    test_sentences,
    train_pos,
    test_pos,
    train_tag,
    test_tag
) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

train_dataset = dataset.EntityDataset(texts = train_sentences, pos = train_pos, tag = train_tag)
valid_dataset = dataset.EntityDataset(texts=test_sentences, pos=test_pos, tag=test_tag)
print('Lengths of dataset: Train {}, Test {} '.format(len(train_dataset), len(valid_dataset)))


# Test tokenization
# a = train_dataset[32]
# ex2_text = ' '.join(a['text'])
# tokenized_ex2_text = config.TOKENIZER.tokenize(ex2_text)
# print(ex2_text)
# print(tokenized_ex2_text)
# print(config.TOKENIZER.convert_tokens_to_ids(tokenized_ex2_text))
# print(a['ids'])
# print(a['target_tag'])

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1)

model = model.EntityModel(num_tag = num_tag, num_pos = num_pos)
print('** GPU ** {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU available: Please train on GPU")

print(device)
model = model.to(device)
#print(list(model.named_parameters()))

num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
print('number of steps {}'.format(num_train_steps))

param_optimizer = list(model.named_parameters())
# Parameters EXCLUDED from optimization
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

# excluded_list = [n for n, p in param_optimizer if any(nd in n for nd in no_decay)]
# print("excluded_list {}".format(excluded_list))

optimizer_parameters = [
  {
    # If parameter NOT in bias/norm weight is non-zero
    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    "weight_decay": 0.001, # Non Bias/Norm are non-zero
  },
  {
    # If parameter in bias/norm weight is ZERO
    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    "weight_decay": 0.0
  },
]

optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

#best_loss = np.Inf
#for epoch in range(config.EPOCHS):
#    train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
#    #test_loss = engine.eval_fn(valid_data_loader, model, device)
#





