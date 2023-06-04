import datetime
import math

import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader


def attention_mask_creator(input_ids):
    attention_masks = []
    for sent in input_ids:
        segments_ids = [int(t > 0) for t in sent]
        attention_masks.append(segments_ids)
    return torch.tensor(attention_masks)


def statistics(group1, group2):
    print('Group 1:', group1.describe())
    print('Group 2:', group2.describe())

    dif = group1.sub(group2, fill_value=0)

    SW_stat, SW_p = stats.shapiro(dif)
    print(SW_stat, SW_p)

    if SW_p >= 0.05:
        print('T-Test:')
        statistic, p = stats.ttest_rel(group1, group2)
    else:
        print('Wilcoxon Test:')
        statistic, p = stats.wilcoxon(group1, group2)

    print('Statistic: {}, p: {}'.format(statistic, p))

    effect_size = statistic / np.sqrt(len(group1))
    print('effect size r: {}'.format(effect_size))


def tokenize_to_id(sentences, tokenizer):
    return [tokenizer.encode(sent) for sent in sentences]


def input_pipeline(sequence, tokenizer, MAX_LEN):

    input_ids = tokenize_to_id(sequence, tokenizer)

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.mask_token_id,
                              truncating="post", padding="post")
    input_ids = torch.tensor(input_ids)

    attention_masks = attention_mask_creator(input_ids)

    return input_ids, attention_masks


def prob_with_prior(pred_TM, pred_TAM, input_ids_TAM, original_ids, tokenizer):
    pred_TM = pred_TM.cpu()
    pred_TAM = pred_TAM.cpu()
    input_ids_TAM = input_ids_TAM.cpu()

    probs = []

    for doc_idx, id_list in enumerate(input_ids_TAM):
        mask_indices = np.where(id_list == tokenizer.mask_token_id)[0]
        target_id = original_ids[doc_idx][mask_indices[0]]
        target_prob = pred_TM[mask_indices[0]][target_id].item()
        prior = pred_TAM[mask_indices[0]][target_id].item()
        probs.append(np.log(target_prob / prior))
    return probs


def model_evaluation(eval_df, tokenizer, model, device):
    max_len = max([len(sent.split()) for sent in eval_df.Sent_TM])
    pos = math.ceil(math.log2(max_len))
    max_len_eval = int(math.pow(2, pos))

    print('max_len evaluation: {}'.format(max_len_eval))

    eval_tokens_TM, eval_attentions_TM = input_pipeline(eval_df.Sent_TM,
                                                        tokenizer,
                                                        max_len_eval)
    eval_tokens_TAM, eval_attentions_TAM = input_pipeline(eval_df.Sent_TAM,
                                                          tokenizer,
                                                          max_len_eval)
    eval_tokens, _ = input_pipeline(eval_df.Sentence, tokenizer, max_len_eval)

    assert eval_tokens_TM.shape == eval_attentions_TM.shape
    assert eval_tokens_TAM.shape == eval_attentions_TAM.shape

    eval_batch = 20
    eval_data = TensorDataset(eval_tokens_TM, eval_attentions_TM,
                              eval_tokens_TAM, eval_attentions_TAM,
                              eval_tokens)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    model.to(device)

    model.eval()
    associations_all = []
    for step, batch in enumerate(eval_dataloader):
        b_input_TM = batch[0].to(device)
        b_att_TM = batch[1].to(device)
        b_input_TAM = batch[2].to(device)
        b_att_TAM = batch[3].to(device)
        with torch.no_grad():
            outputs_TM = model(b_input_TM,
                               attention_mask=b_att_TM)

            outputs_TAM = model(b_input_TAM,
                                attention_mask=b_att_TAM)

            predictions_TM = softmax(outputs_TM[0], dim=None)
            predictions_TAM = softmax(outputs_TAM[0], dim=None)

        assert predictions_TM.shape == predictions_TAM.shape

        associations = prob_with_prior(predictions_TM,
                                       predictions_TAM,
                                       b_input_TAM,
                                       batch[4],
                                       tokenizer)

        associations_all += associations

    return associations_all


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))
