from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from src.utils.imdb_processor import IMDBProcessor
from src.utils.input_processor import InputBatch
from src.utils.my_logger import logger
from src.transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from src.transformer.modeling import TinyBertForSequenceClassification
from src.transformer import BertAdam
from src.transformer import BertTokenizer


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        if type(result) == str:
            writer.write("%s\n" % (result))
            return
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    return parser.parse_args()


class KnowledgeDistillation:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.seed_initialize()
        self.data_processor = IMDBProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(self.args.student_model,
                                                       do_lower_case=self.args.do_lower_case)

        self.student_model = TinyBertForSequenceClassification.from_pretrained(self.args.student_model,
                                                                               num_labels=2)
        self.student_model.to(self.device)

        if self.n_gpu > 1:
            self.student_model = torch.nn.DataParallel(self.student_model)
        if not self.args.do_eval:
            self.teacher_model = TinyBertForSequenceClassification.from_pretrained(self.args.teacher_model,
                                                                                   num_labels=2)
            self.teacher_model.to(self.device)
            if self.n_gpu > 1:
                self.teacher_model = torch.nn.DataParallel(self.teacher_model)

    def seed_initialize(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

        # Prepare task settings
        if os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.args.output_dir))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

    def prepare_train_data(self):
        label_list = self.data_processor.get_labels()
        train_examples = self.data_processor.get_train_examples(self.args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / self.args.train_batch_size) * self.args.num_train_epochs
        train_batch = InputBatch(train_examples, label_list, self.args.max_seq_length, self.tokenizer)
        train_data, _ = train_batch.get_tensor_data()
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)
        return train_dataloader, num_train_optimization_steps

    def prepare_eval_data(self):
        eval_examples = self.data_processor.get_dev_examples(self.args.data_dir)
        label_list = self.data_processor.get_labels()
        eval_batch = InputBatch(eval_examples, label_list, self.args.max_seq_length, self.tokenizer)
        eval_data, eval_labels = eval_batch.get_tensor_data()
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        return eval_dataloader, eval_labels

    def evaluate(self):

        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_dataloader, eval_labels = self.prepare_eval_data()

        self.student_model.eval()
        result = self.calculate_loss(eval_dataloader, eval_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    def get_optimizer(self, num_train_optimization_steps):
        param_optimizer = list(self.student_model.named_parameters())
        size = 0
        for n, p in self.student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not self.args.pred_distill:
            schedule = 'none'
        return BertAdam(optimizer_grouped_parameters,
                        schedule=schedule,
                        lr=self.args.learning_rate,
                        warmup=self.args.warmup_proportion,
                        t_total=num_train_optimization_steps)

    @staticmethod
    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def train(self):
        train_dataloader, num_train_optimization_steps = self.prepare_train_data()
        eval_dataloader, eval_labels = self.prepare_eval_data()
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", self.args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        optimizer = self.get_optimizer(num_train_optimization_steps)
        # Prepare loss functions
        loss_mse = MSELoss()

        # Train and evaluate
        global_step = 0
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            self.student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != self.args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.

                student_logits, student_atts, student_reps = self.student_model(input_ids,
                                                                                segment_ids,
                                                                                input_mask,
                                                                                is_student=True)

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = self.teacher_model(input_ids, segment_ids, input_mask)

                if not self.args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                else:
                    cls_loss = self.soft_cross_entropy(student_logits / self.args.temperature,
                                                       teacher_logits / self.args.temperature)
                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            self.student_model.eval()
            result = self.calculate_loss(eval_dataloader, eval_labels)
            result_to_file("###### epoch {} results are \n".format(epoch_), output_eval_file)
            result_to_file(result, output_eval_file)
            model_to_save = self.student_model.module if hasattr(self.student_model, 'module') else self.student_model
            model_name = WEIGHTS_NAME
            output_model_file = os.path.join(self.args.output_dir, model_name)
            output_config_file = os.path.join(self.args.output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_vocabulary(self.args.output_dir)
            self.student_model.train()

    def calculate_loss(self, eval_dataloader, eval_labels):
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
            batch_ = tuple(t.to(self.device) for t in batch_)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

                logits, _, _ = self.student_model(input_ids, segment_ids, input_mask)

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        result = self.compute_metrics(preds, eval_labels.numpy())
        result['eval_loss'] = eval_loss

        return result

    @staticmethod
    def compute_metrics(preds, labels):
        return {
            'acc': (preds == labels).mean(),
            'recall': recall_score(labels, preds),
            'precision': precision_score(labels, preds),
            'f1_score': f1_score(labels, preds),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'confusion_matrix': confusion_matrix(labels, preds)
        }

    def __call__(self):
        if self.args.do_eval:
            self.evaluate()
        else:
            self.train()


if __name__ == '__main__':
    args = get_arguments()
    kd = KnowledgeDistillation(args)
    kd()
