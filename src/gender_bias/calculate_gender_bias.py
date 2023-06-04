import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rtpt import RTPT
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm

from src.transformer.modeling import BertForSequenceClassification, TinyBertForSequenceClassification
from src.transformer.tokenization import BertTokenizer
from src.utils.input_processor import InputExample, InputFeatures


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def create_examples(name):
    input_file = os.path.join(os.getcwd(), name)
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
    examples = []
    for (i, line) in enumerate(lines[1:]):
        examples.append(InputExample(
            guid=f"test-{i}",
            text_a=line[1],
            label=line[2])
        )
    return examples


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_to_features(examples,
                        max_seq_length,
                        tokenizer):
    label_map = {
        '1': 1,
        '0': 0
    }

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        elif len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label in label_map:
            label_id = label_map[example.label]

            if ex_index < 1:
                print('logging here')

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              seq_length=seq_length)
            )
    return features


def calc_gender_bias(task="IMDB", model_id="tinybert"):
    df_l = pd.read_pickle('{}_l_test'.format(task))

    for elem in ['pro', 'weat', 'all']:
        male_file = f'text_{elem}_M'
        female_file = f'text_{elem}_F'

        df_exp = df_l[['ID', male_file, female_file, 'label']]
        df_exp['label'][df_exp['label'] == 'pos'] = 1
        df_exp['label'][df_exp['label'] == 'neg'] = 0

        male_df_exp = df_exp[[male_file, 'label']]
        female_df_exp = df_exp[[female_file, 'label']]

        male_df_exp = male_df_exp.rename(columns={male_file: 'text', 'label': 'label'})
        female_df_exp = female_df_exp.rename(columns={female_file: 'text', 'label': 'label'})

        male_df_exp.to_csv('text_{}_M.tsv'.format(elem), sep="\t")
        female_df_exp.to_csv('text_{}_F.tsv'.format(elem), sep="\t")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = f'models/{model_id}'
        if model_id == 'bertbase':
            model = BertForSequenceClassification.from_pretrained(path, num_labels=2)
        else:
            model = TinyBertForSequenceClassification.from_pretrained(path, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)

        model.to(device)

        model.eval()
        batch_size = 32
        raw_pred_m = None
        raw_pred_f = None
        for gender in ['M', 'F']:
            examples = create_examples(name=f'text_{elem}_{gender}.tsv')
            features = convert_to_features(examples, 512, tokenizer)
            data, labels = get_tensor_data(features)
            seq_sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=seq_sampler, batch_size=batch_size)

            preds = []
            for step, batch in enumerate(tqdm(dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if model_id == 'bertbase':
                    logits = model(input_ids, segment_ids, input_mask)
                else:
                    logits, atts, reps = model(input_ids, segment_ids, input_mask, is_student=False)
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                preds.append(predictions.detach().cpu().numpy())
            if gender == 'M':
                raw_pred_m = (np.concatenate(preds, axis=0))
            else:
                raw_pred_f = (np.concatenate(preds, axis=0))

        y_soft_m = F.softmax(torch.from_numpy(raw_pred_m), dim=1).tolist()
        y_soft_f = F.softmax(torch.from_numpy(raw_pred_f), dim=1).tolist()

        m_soft = [e[0] for e in y_soft_m]
        f_soft = [e[0] for e in y_soft_f]

        m_soft = m_soft + [0 for _ in range(25000 - len(m_soft))]
        f_soft = f_soft + [0 for _ in range(25000 - len(f_soft))]
        df_exp['pos_prob_m'] = m_soft
        df_exp['pos_prob_f'] = f_soft
        df_exp['bias'] = df_exp['pos_prob_m'] - df_exp['pos_prob_f']

        df_exp.to_pickle('results/rating_{}_{}_{}'.format(task, model_id, elem))
        print('saved in df')
    return df_exp


def get_tensor_data(features):
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


if __name__ == '__main__':
    specs = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]
    rtpt_train = RTPT(name_initials='SJ', experiment_name="IMDB", max_iterations=len(specs) * 2)
    rtpt_train.start()

    rate_output = calc_gender_bias()
    print(rate_output)
