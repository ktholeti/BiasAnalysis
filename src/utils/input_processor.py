import torch
from torch.utils.data import TensorDataset

from my_logger import logger


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class InputBatch:

    def __init__(self, examples, label_list, max_seq_length, tokenizer):
        self.examples = examples
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def _convert_examples_to_features(self):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label: i for i, label in enumerate(self.label_list)}

        features = []
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(self.examples)))

            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            seq_length = len(input_ids)

            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            label_id = label_map[example.label]

            if ex_index < 1:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: {}".format(example.label))
                logger.info("label_id: {}".format(label_id))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              seq_length=seq_length))
        return features

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def get_tensor_data(self):
        features = self._convert_examples_to_features()
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_label_ids, all_seq_lengths)
        return tensor_data, all_label_ids
