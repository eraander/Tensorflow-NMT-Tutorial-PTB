#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Entry Point

This project implements `Grammar as a Foreign Language` by Vinyals et al. (2014)
https://arxiv.org/abs/1412.7449

Feel free to change any part of this code
"""
import argparse
import os
import pickle
import time


__author__ = 'Jayeol Chun, Erik Andersen'


argparser = argparse.ArgumentParser("PA4 Argparser")

# paths
argparser.add_argument(
  '--data_dir', default='./data', help='path to data directory')
argparser.add_argument(
  '--model_dir', default='./attention', help='path to output directory')
argparser.add_argument(
  '--evalb_dir', default='./EVALB', help='path to EVALB directory')

# preprocessing flags
argparser.add_argument(
  '--enc_max_len', default=-1, type=int,
  help='how many encoder tokens to keep. -1 for max length from corpus')
argparser.add_argument(
  '--dec_max_len', default=-1, type=int,
  help='how many decoder tokens to keep. -1 for max length from corpus')
argparser.add_argument(
  '--do_reverse', action='store_true', help='whether to reverse sents')
argparser.add_argument(
  '--do_lower', action='store_true', help='whether to lower-case sents')

argparser.add_argument(
  '--src', default='.sts')

argparser.add_argument(
  '--tgt', default='.lnr')

def run_evalb(evalb_dir, gold_path, pred_path):
  """executes evalb automatically

  Assumed that `EVALB` is installed through `make` command
  """
  import sys
  import subprocess

  if not os.path.exists(pred_path):
    print(
      "[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(pred_path))
    sys.exit(-1)

  evalb = os.path.join(evalb_dir, 'evalb')
  error_flag = '-e'
  num_error_trees = 10000 # arbitrarily big
  command = "{} {} {} {} {}".format(
    evalb, error_flag, num_error_trees, gold_path, pred_path)

  print("Running EVALB with command:", command)
  proc = subprocess.Popen(
    command, stdout=sys.stdout, stderr=sys.stderr, shell=True,
    universal_newlines=True)
  proc.wait()

def maybe_resolve_path_conflict(path):
  """creates a unique filename incase `path` already exists"""
  base_dir, filename = os.path.split(path)

  sep = filename.index('.')
  name, ext = filename[:sep], filename[sep:]

  while os.path.exists(path):
    if "_" in name:
      name_split = name.split("_")
      print(name_split)
      i = int(name_split[-1])
      name = name_split[0] + f"_{i+1}"
    else:
      name += "_1"

    path = os.path.join(base_dir, name + ext)

  return path

def export(data, out_path):
  out_path = maybe_resolve_path_conflict(out_path)
  with open(out_path, 'w') as f:
    f.write("\n".join(data))
  return out_path

def postprocess(data, sentences):
  """TODO: implement a post-processing function

  Postprocesses the data for use with evalb. Weaves
  the words from the sentences back into the tagged sentences.

  Args:
    data: list of str
    sentences: list of str

  Returns:
    cured_data: list of str

  """
  cured_data = []
  for i, datum in enumerate(data):
    cured_datum = []
    sentence_split = sentences[i].split()
    datum_split = datum.split()
    unbalanced_count = 0
    right_bracket_count = 0
    sentence_index = 0
    for tok in datum_split:
      if sentence_index >= len(sentence_split)-1: break
      if tok == '<end>': continue
      elif '(' in tok:
        if right_bracket_count > 0:
          right_brackets = ')' * right_bracket_count
          cured_datum.append((sentence_split[sentence_index] + right_brackets))
          sentence_index += 1
        right_bracket_count = 0
        cured_datum.append(tok)
        unbalanced_count += 1
      else:
        if unbalanced_count == 0: continue
        else:
          right_bracket_count += 1
          unbalanced_count -= 1
    if right_bracket_count > 0:
      right_brackets = ')' * right_bracket_count
      if sentence_index < len(sentence_split):
        cured_datum.append((sentence_split[sentence_index] + right_brackets))
        sentence_index += 1
      else:
        cured_datum.append(right_brackets)
    while sentence_index < len(sentence_split):
      cured_datum.append('(X')
      cured_datum.append((sentence_split[sentence_index] + ')'))
      sentence_index += 1
    cured_datum.append(')' * unbalanced_count)
    cured_datum = ' '.join(cured_datum)
    cured_data.append(cured_datum)
  return cured_data

def main(args):
  """
  main method
  """
  begin = time.time()

  dev_output_filepath = os.path.join(args.model_dir, "output_dev")
  test_output_filepath = os.path.join(args.model_dir, "output_test")
  dev_source_filename = "dev" + args.src
  dev_source_filepath = os.path.join(args.data_dir, dev_source_filename)
  test_source_filename = "test" + args.src
  test_source_filepath = os.path.join(args.data_dir, test_source_filename)
  dev_target_filename = "dev" + args.tgt
  dev_target_filepath = os.path.join(args.data_dir, dev_target_filename)
  test_target_filename = "test" + args.tgt
  test_target_filepath = os.path.join(args.data_dir, test_target_filename)

  # open sentences and trees for reading
  with open(dev_output_filepath, "r") as output_dev_f:
    dev_preds = output_dev_f.readlines()

  with open(test_output_filepath, "r") as output_test_f:
    test_preds = output_test_f.readlines()

  with open(dev_source_filepath, "r") as sents_dev_f:
    dev_sents = sents_dev_f.readlines()

  with open(test_source_filepath, "r") as sents_test_f:
    test_sents = sents_test_f.readlines()

  with open(dev_target_filepath, "r") as target_dev_f:
    dev_golds = target_dev_f.readlines()

  with open(test_target_filepath, "r") as target_test_f:
    test_golds = target_test_f.readlines()


  # we will export `preds` and `golds` before post-processing, in case you want
  # to experiment with different postprocessing methods separately
  print("Exporting golds and preds..")

  dev_pred_path = './devpreds.out'
  dev_pred_path = export(dev_preds, dev_pred_path)

  dev_gold_path = './devgolds.out'
  dev_gold_path = export(dev_golds, dev_gold_path)

  test_pred_path = './testpreds.out'
  test_pred_path = export(test_preds, test_pred_path)

  test_gold_path = './testgolds.out'
  test_gold_path = export(test_golds, test_gold_path)

  print("Postprocessing..")
  dev_preds = postprocess(dev_preds, dev_sents)
  dev_golds = postprocess(dev_golds, dev_sents)
  test_preds = postprocess(test_preds, test_sents)
  test_golds = postprocess(test_golds, test_sents)

  print("Exporting postprocessed golds and preds..")
  dev_gold_pp_path = dev_gold_path + '.pp'
  dev_gold_pp_path = export(dev_golds, dev_gold_pp_path)

  dev_pred_pp_path = dev_pred_path + '.pp'
  dev_pred_pp_path = export(dev_preds, dev_pred_pp_path)

  test_gold_pp_path = test_gold_path + '.pp'
  test_gold_pp_path = export(test_golds, test_gold_pp_path)

  test_pred_pp_path = test_pred_path + '.pp'
  test_pred_pp_path = export(test_preds, test_pred_pp_path)

  # automatic execution of EVALB script
  run_evalb(args.evalb_dir, gold_path=dev_gold_pp_path, pred_path=dev_pred_pp_path)
  run_evalb(args.evalb_dir, gold_path=test_gold_pp_path, pred_path=test_pred_pp_path)

  print("Execution Time: {:.2f}s".format(time.time() - begin))

if __name__ == '__main__':
  args = argparser.parse_args()

  print("******** FLAGS ********")

  main(args)
