#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Pre-processing
  This code contains pre-processing code to prepare Penn Treebank data for use with tensorflow/nmt.
"""
import os
import argparse

import nltk
import tensorflow as tf
import tqdm, re

__author__ = 'Jayeol Chun, Erik Andersen'

argparser = argparse.ArgumentParser("PA4 Argparser")

# paths
argparser.add_argument(
  '--data_dir', default='./data', help='path to data directory')

argparser.add_argument(
  '--src', default='.sts')

argparser.add_argument(
  '--tgt', default='.lnr')

argparser.add_argument(
  '--do_reverse', default=False)

##################################### Util #####################################
def split_corpus(data, sep_a, sep_b):
  """splits into train, dev and test using `sep_a` and `sep_b`

  Args:
    data: array, output of `tokenize`
    sep_a: int, index separating train and dev
    sep_b: int, index separating dev and test

  Returns:
    tuple, (train, dev, test)
  """
  train = data[:sep_a]
  dev = data[sep_a:-sep_b]
  test = data[-sep_b:]
  return train, dev, test

def fit_tokenizer(data, unk=None, lower=False):
  """fits the data into a tensorflow Tokenizer object

  Args:
    data: list of str
    unk: str, which token to use for oov_token
    lower: bool, whether to lower-case string

  Returns:
    tf.keras.preprocessing.text.Tokenizer object
  """
  tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='', oov_token=unk, lower=lower)
  tokenizer.fit_on_texts(data)
  return tokenizer

def linearize_parse_tree(tree):
  """linearizes a tree into a sequence

  Deletes the '-NONE-' tag, removes extra numbers from the end of tags, and adds
  a tag to the right bracket for parallelism. Removes all words for a manageable
  vocabulary size.

  Args:
    tree: nltk.tree.Tree object

  Returns:
    linearized list of tokens
  """
  tree = tree.__str__().replace("\n","").split()

  out = []
  parenthesis_stack = []
  for i, tok in enumerate(tree):
    if '(' in tok:
      tok = re.sub("[-|=][0-9]+", "", tok)
      if '-NONE-' not in tok:
        out.append(tok)
      parenthesis_stack.append(tok)
    else:
      idx = tok.index(")")
      for _ in range(idx, len(tok)):
        new_tok = parenthesis_stack.pop()
        right_tok = ')' + new_tok[1:]
        if '-NONE-' not in right_tok:
          out.append(right_tok)
  return out

#################################### Loader ####################################
def load_dataset(dataset_dir):
  """loads a single dataset (train, dev, test)

  Note that technically this is not loading, which happens when we iterate
    through the CorpusView objects later.

  Args:
    dataset_dir: str, path to root of a single dataset in PTB

  Returns:
    tuple, ConcatenatedCorpusView objects corresponding to raw sentences and
      parse trees
  """
  reader = nltk.corpus.BracketParseCorpusReader(dataset_dir, r'.*/wsj_.*\.mrg')
  sents = reader.sents()
  trees = reader.parsed_sents()
  return sents, trees

def load_data(data_dir):
  """loads Penn TreeBank data

  Args:
    data_dir: str, path to root of PTB

  Returns:
    dict, where keys are 'dev', 'train', 'test' and values are return values
      from `load_dataset` above
  """
  data = {}
  datasets = ['dev', 'train', 'test']
  for dataset in datasets:
    dataset_dir = os.path.join(data_dir, dataset)
    data[dataset] = load_dataset(dataset_dir)
  return data

################################# Preprocessor #################################
def preprocess(data_dir, do_reverse, do_lower=False):
  """loads and preprocesses PTB

  1. loads data with NLTK
  2. performs a preliminary processing on both sentences and parse trees
  3. collects vocab
  4. tokenize and convert strings into indices
  5. groups train, dev and test data

  Args:
    data_dir: str, path to root of PTB
    do_reverse: bool, whether to reverse encoder-input sentences
    do_lower: bool, whether to lower-case

  Returns:
    tuple
  """

  if not os.path.exists(data_dir):
    raise ValueError(f"{data_dir} doesn't exist in `preprocess`")

  # 1. loads data
  data = load_data(data_dir)

  # 2. preliminary processing
  for dataset, datum in data.items():
    print(f"Loading {dataset}..")
    sents, trees = datum

    # actual loading of dataset
    _sents = []
    for sent in tqdm.tqdm(sents):
      new_sent = [word for word in sent if not "*" in word]
      if do_reverse:
        new_sent = list(reversed(new_sent))
      _sents.append(new_sent)

    _trees, _labels, _lin_trees = [], [], []
    for tree in tqdm.tqdm(trees):
      lin_tree = linearize_parse_tree(tree)
      _trees.append(['<end>'] + lin_tree)
      _labels.append(lin_tree + ['<end>'])
      _lin_trees.append(" ".join(lin_tree)) # for evaluating test

    # override original data
    data[dataset] = (_sents, _trees, _labels, _lin_trees)

    print("Sample data from", dataset)
    print("\tSent:", _sents[0])
    print("\tTree:", _trees[0])
    print("\tLabel:", _labels[0])

  # 3. collect vocab from train and dev set only
  train_dev_sents = data['train'][0] + data['dev'][0]
  train_dev_trees = data['train'][1] + data['dev'][1]

  # fit the data to Tensorflow; we will add <unk> later
  enc_tokenizer = fit_tokenizer(
    train_dev_sents, lower=do_lower)
  dec_tokenizer = fit_tokenizer(train_dev_trees)

  # + 1 for padding, which gets the index 0
  enc_vocab_size = len(enc_tokenizer.word_index) + 1
  dec_vocab_size = len(dec_tokenizer.word_index) + 1

  # just so the user can see how large the vocab size is
  print("\nEncoder Vocab Size:", enc_vocab_size)
  print("Decoder Vocab Size:", dec_vocab_size)
  print("Decoder Vocabs:", dec_tokenizer.word_index.keys())
  vocab_source_filename = "vocab" + args.src
  vocab_source_filepath = os.path.join(data_dir, vocab_source_filename)
  vocab_target_filename = "vocab" + args.tgt
  vocab_target_filepath = os.path.join(data_dir, vocab_target_filename)

  with open(vocab_source_filepath, "w") as source_f:
    source_f.write("<unk>\n")
    source_f.write("<s>\n")
    source_f.write("</s>\n")
    for key in enc_tokenizer.word_index.keys():
      source_f.write(key + "\n")

  with open(vocab_target_filepath, "w") as target_f:
    target_f.write("<unk>\n")
    target_f.write("<s>\n")
    target_f.write("</s>\n")
    for key in dec_tokenizer.word_index.keys():
      target_f.write(key + "\n")

  # 4. tokenize and convert to indices all together
  all_sents = train_dev_sents + data['test'][0]
  all_trees = train_dev_trees + data['test'][1]
  all_labels = data['train'][2] + data['dev'][2] + data['test'][2]

  train_len = len(data['train'][0])
  dev_len = len(data['dev'][0])
  test_len = len(data['test'][0])

  enc_inputs_train, enc_inputs_dev, enc_inputs_test = split_corpus(
    all_sents, train_len, test_len)
  dec_inputs_train, dec_inputs_dev, dec_inputs_test = split_corpus(
    all_trees, train_len, test_len)
  dec_outputs_train, dec_outputs_dev, dec_outputs_test = split_corpus(
    all_labels, train_len, test_len)

  train_dataset = [enc_inputs_train, dec_inputs_train, dec_outputs_train]
  dev_dataset = [enc_inputs_dev, dec_inputs_dev, dec_outputs_dev]
  test_dataset = [enc_inputs_test, dec_inputs_test, dec_outputs_test]

  return train_dataset, dev_dataset, test_dataset

if __name__ == "__main__":
  args = argparser.parse_args()
  data_dir = args.data_dir
  train_data, dev_data, test_data = preprocess(data_dir, args.do_reverse)

  train_source_filename = "train" + args.src
  train_source_filepath = os.path.join(data_dir, train_source_filename)
  train_target_filename = "train" + args.tgt
  train_target_filepath = os.path.join(data_dir, train_target_filename)
  dev_source_filename = "dev" + args.src
  dev_source_filepath = os.path.join(data_dir, dev_source_filename)
  dev_target_filename = "dev" + args.tgt
  dev_target_filepath = os.path.join(data_dir, dev_target_filename)
  test_source_filename = "test" + args.src
  test_source_filepath = os.path.join(data_dir, test_source_filename)
  test_target_filename = "test" + args.tgt
  test_target_filepath = os.path.join(data_dir, test_target_filename)

  with open(train_source_filepath, "w") as source_train_f:
    enc_train = train_data[0]
    for datum in enc_train:
      source_train_f.write(' '.join(datum) + "\n")

  with open(train_target_filepath, "w") as target_train_f:
    dec_train = train_data[1]
    for datum in dec_train:
      target_train_f.write(' '.join(datum) + "\n")

  with open(dev_source_filepath, "w") as source_dev_f:
    enc_dev = dev_data[0]
    for datum in enc_dev:
      source_dev_f.write(' '.join(datum) + "\n")

  with open(dev_target_filepath, "w") as target_dev_f:
    dec_dev = dev_data[1]
    for datum in dec_dev:
      target_dev_f.write(' '.join(datum) + "\n")
      
  with open(test_source_filepath, "w") as source_test_f:
    enc_test = test_data[0]
    for datum in enc_test:
      source_test_f.write(' '.join(datum) + "\n")

  with open(test_target_filepath, "w") as target_test_f:
    dec_test = test_data[1]
    for datum in dec_test:
      target_test_f.write(' '.join(datum) + "\n")

