"""Train the model"""

import argparse
import gensim
import logging
import os
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.input_fn import load_signal
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments/base_model")
parser.add_argument('--data_dir', default='data')
parser.add_argument("--restore_dir", default=None)
parser.add_argument("--word2vec_file", default="GoogleNews-vectors-negative300.bin")

if __name__ == '__main__':
    tf.set_random_seed(230)
    args = parser.parse_args()
    
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path)

    params.evaluate()

    path_symbols = os.path.join(args.data_dir, "symbols.txt")
    path_train = os.path.join(args.data_dir, "train")
    path_dev = os.path.join(args.data_dir, "dev")
    path_corpus = os.path.join(args.data_dir, "corpus")

    set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading pretrained Word2Vec...")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)
    logging.info("- done.")

    logging.info("Building datasets...")
    with open(os.path.join(args.data_dir, "symbols.txt")) as f_symbols:
        symbols = [line.strip() for line in f_symbols]
    train_signal_map = {symbol: load_signal(path_train, path_corpus, symbol) for symbol in symbols[:1]}
    dev_signal_map = {symbol: load_signal(path_dev, path_corpus, symbol) for symbol in symbols[:1]}
    params.eval_size = params.dev_size
    params.buffer_size = params.train_size
    train_inputs = input_fn("train", train_signal_map, word2vec, params)
    eval_inputs = input_fn("eval", dev_signal_map, word2vec, params)
    logging.info("- done.")
    
    logging.info("Creating the model...")
    train_model_spec = model_fn("train", train_inputs, params)
    eval_model_spec = model_fn("eval", eval_inputs, params, reuse=True)
    logging.info("- done.")

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)
