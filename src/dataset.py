import torch
import config

class EntityDataset():
    def __init__(self, texts, pos, tag):
        self.texts = texts
        self.pos = pos
        self.tag = tag

    def __len__(self):
        return len(self.texts)

    def __str__(self):
        return str(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tag = self.tag[item]

        # Tokenize for BERT ONE list at a time
        ids = []
        target_pos = []
        target_tag = []
        for i, s in enumerate(text):
            # Each word in text is tokenized and can be split into multiple tokens
            input = config.TOKENIZER.encode(
                     s,
                     add_special_tokens = False) # since this is within sentence do not add CLS and SEP tokens
            ids.extend(input)
            input_len = len(input)
            target_pos.extend([pos[i]] * input_len) # since each word can be tokenized into multiple tokens
            target_tag.extend([tag[i]] * input_len)

      # PAD to max_len and leave 2 for special tokens
        ids = ids[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        # ADD special tokens 101=CLS ; 102 = SEP
        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]
        mask = [1] * len(ids) # attention mask
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids) # pad short sentences
        ids = ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
                'ids' : torch.tensor(ids, dtype = torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'target_pos': torch.tensor(target_pos, dtype=torch.long),
                'target_tag': torch.tensor(target_tag, dtype=torch.long)
                }