import torch
import transformers
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
NCBI_TRAINING_FILE = '../input/NCBITraining_tagged_revised.csv'

MAX_LEN = 128 # Padding length
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
TOKENIZER =  AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") # Clinical Bert
MODEL_PATH = './model.bin'

