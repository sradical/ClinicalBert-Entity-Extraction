import torch
import transformers
import spacy
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
NCBI_TRAINING_FILE = '../input/NCBITraining_tagged.pkl'
Kaggle_TRAINING_FILE = '../input/KaggleNERDataset.pkl'

MAX_LEN = 128 # Padding length
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10

# Clinical Bert
BASE_MODEL = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True, return_dict=False)
TOKENIZER =  AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
MODEL_PATH = './model.bin'

# Bert-Base-uncased
#transformer_model = BertModel.from_pretrained("bert-base-uncased")
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
#MODEL_PATH = './kaggle_model.bin'