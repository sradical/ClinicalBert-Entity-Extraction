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
#TOKENIZER =  AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") # Clinical Bert
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
#MODEL_PATH = './model.bin'
MODEL_PATH = './kaggle_model.bin'