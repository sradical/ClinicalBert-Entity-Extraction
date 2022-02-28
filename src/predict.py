import joblib
import torch.hub

import config
import dataset
import model
import engine
from model import EntityModel

if __name__ == "__main__":
    meta_data = joblib.load('meta.bin')

    enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    print('#Tag {} #Pos {}'.format(num_tag, num_pos))
    print('Tag {}'.format(enc_tag.classes_))
    print('Pos {}'.format((enc_pos.classes_)))

    sentence = """The risk of cancer, especially lymphoid neoplasias, is substantially elevated in A-T patients and has long been associated with chromosomal instability."""
    tokenized_sentence = config.TOKENIZER(sentence)
    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    model = model.EntityModel(num_tag=num_tag, num_pos=num_pos)

    test_dataset = dataset.EntityDataset(
        texts = [sentence],
        pos = [[0] * len(sentence)],
        tag = [[0] * len(sentence)]
    )


