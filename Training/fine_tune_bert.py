# This is a script for fine-tuning bert for classification

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from early_stopper import EarlyStopper
from imdb_processor import IMDBProcessor
from input_processor import InputBatch
from my_logger import logger
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import BertForSequenceClassification
from transformer.optimization import BertAdam
from transformer.tokenization import BertTokenizer


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
    parser.add_argument("--pre_trained_bert",
                        default=None,
                        type=str,
                        required=True,
                        help="The pre-trained bert model dir.")
    parser.add_argument("--task_name",
                        default="IMDB",
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--eval_mode",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-2,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--eval_step',
                        type=int,
                        default=10)

    return parser.parse_args()


class FineTuneBert:

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.seed_initialize()
        self.data_processor = IMDBProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(self.args.pre_trained_bert,
                                                       do_lower_case=self.args.do_lower_case)

        self.model = BertForSequenceClassification.from_pretrained(self.args.pre_trained_bert,
                                                                   num_labels=2)
        self.model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def __call__(self):
        if self.args.eval_mode:
            eval_dataloader, eval_labels = self.prepare_eval_data()
            self.evaluate(eval_dataloader)
        else:
            self.train()

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

    def train(self):
        train_dataloader, num_train_optimization_steps = self.prepare_train_data()
        eval_dataloader, eval_labels = self.prepare_eval_data()
        optimizer = self.get_optimizer(num_train_optimization_steps)
        num_labels = len(self.data_processor.get_labels())

        global_step = 0
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        loss_func = CrossEntropyLoss(weight=torch.tensor([8.3, 1.]).to(self.device))
        early_stopper = EarlyStopper()
        for epoch_ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            self.model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != self.args.train_batch_size:
                    continue
                logits = self.model(input_ids, segment_ids, input_mask)
                loss = loss_func(logits.view(-1, num_labels), label_ids.view(-1))

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

            # evaluating after epoch
            result = self.evaluate(eval_dataloader)
            should_stop = early_stopper.early_stop(result['eval_loss'])
            if should_stop:
                break
            if early_stopper.counter == 0:
                logger.info("***** Saving best model till now *****")
                result_to_file("###### saving the model at epoch {} \n".format(epoch_), output_eval_file)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_name = WEIGHTS_NAME
                output_model_file = os.path.join(self.args.output_dir, model_name)
                output_config_file = os.path.join(self.args.output_dir, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                self.tokenizer.save_vocabulary(self.args.output_dir)

            self.model.train()

    def calculate_eval_loss(self, eval_dataloader):
        eval_loss = 0
        nb_eval_steps = 0

        for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
            batch_ = tuple(t.to(self.device) for t in batch_)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
                logits = self.model(input_ids, segment_ids, input_mask)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss(weight=torch.tensor([8.3, 1.]).to(self.device))
            tmp_eval_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        return {'eval_loss': eval_loss}

    def evaluate(self, eval_dataloader):
        self.model.eval()
        result = self.calculate_eval_loss(eval_dataloader)

        output_eval_file = os.path.join(self.args.output_dir, "test_results.txt")
        result_to_file("###### test results are \n", output_eval_file)
        result_to_file(result, output_eval_file)
        return result

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
        if self.args.eval_mode:
            eval_examples = self.data_processor.get_test_examples(self.args.data_dir)
        else:
            eval_examples = self.data_processor.get_dev_examples(self.args.data_dir)
        label_list = self.data_processor.get_labels()
        eval_batch = InputBatch(eval_examples, label_list, self.args.max_seq_length, self.tokenizer)
        eval_data, eval_labels = eval_batch.get_tensor_data()
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        return eval_dataloader, eval_labels

    def get_optimizer(self, num_train_optimization_steps):
        param_optimizer = list(self.model.named_parameters())
        size = 0
        for n, p in self.model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        return BertAdam(optimizer_grouped_parameters,
                        schedule=schedule,
                        lr=self.args.learning_rate,
                        warmup=self.args.warmup_proportion,
                        t_total=num_train_optimization_steps)


if __name__ == "__main__":
    arguments = get_arguments()
    fine_tune_bert = FineTuneBert(arguments)
    fine_tune_bert()
