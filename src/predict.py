import joblib
import torch.hub

import config
import dataset
import model
import engine
from model import EntityModel

if __name__ == "__main__":

    # check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} map_location: {map_location}")
    meta_data = joblib.load('meta.bin')

    enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    #print('#Tag {} #Pos {}'.format(num_tag, num_pos))
    #print('Tag {}'.format(enc_tag.classes_))
    #print('Pos {}'.format((enc_pos.classes_)))

    sentence = """The risk that is skin tumor is elevated"""
    tokenized_sentence = config.TOKENIZER.tokenize(sentence)
    encoded_sentence = config.TOKENIZER.encode(sentence)
    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)
    print(encoded_sentence)

    model = model.EntityModel(config.BASE_MODEL, num_tag=num_tag, num_pos=num_pos )
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=map_location))
    model.to(device)

    test_dataset = dataset.EntityDataset(
        texts = [sentence],
        pos = [[0] * len(sentence)],
        tag = [[0] * len(sentence)]
    )

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)


        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(encoded_sentence)]
        )
        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(encoded_sentence)]
        )




