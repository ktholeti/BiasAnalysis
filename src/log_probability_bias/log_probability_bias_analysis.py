import argparse
import random

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

from log_prob_utils import model_evaluation, statistics
from src.transformer.modeling import BertForMaskedLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', help='.tsv file', required=True)
    parser.add_argument('--out', help='output directory', required=True)
    parser.add_argument('--model', help='which model to use', required=True)
    parser.add_argument('--seed', required=False, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pretrained_model = args.model

    print('-- Prepare evaluation data --')
    eval_data = pd.read_csv(args.eval, sep='\t')

    print('-- Import BERT model --')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    model = BertForMaskedLM.from_pretrained(pretrained_model)

    print('-- Calculate associations--')
    pre_associations = model_evaluation(eval_data, tokenizer, model, device)

    eval_data = eval_data.assign(Pre_Assoc=pre_associations)

    eval_data.to_csv(args.out + '.csv', sep='\t', encoding='utf-8', index=False)

    eval_m = eval_data.loc[eval_data.Prof_Gender == 'male']
    eval_f = eval_data.loc[eval_data.Prof_Gender == 'female']

    statistics(eval_f.Pre_Assoc, eval_m.Pre_Assoc)
    print('Done')
