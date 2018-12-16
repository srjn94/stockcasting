import os
import random
import tensorflow as tf
from model.utils import print_dataset
from datetime import datetime

def load_signal(path, corpus_path, symbol, keepnum=None):
    with open(os.path.join(path, symbol, "dates.txt"), "r") as f_dates:
        dates = [datetime.strptime(line.strip(), "%Y-%m-%d") for line in f_dates]
    with open(os.path.join(path, symbol, "news.txt"), "r") as f_news:
        corpora = []
        for i, line in enumerate(f_news):
            filenames = []
            for name in line.strip().split(" "):
                if name == "":
                    filenames.append(os.path.join(corpus_path, "null.txt"))
                else:
                    filenames.append(os.path.join(corpus_path, dates[i].strftime("%Y/%m/%d"), name))
            corpora.append(filenames)
    with open(os.path.join(path, symbol, "stocks.txt"), "r") as f_stocks:
        stocks = [float(line.strip()) for line in f_stocks]
    if keepnum is not None:
        corpora = corpora[:keepnum]
        stocks = stocks[:keepnum]
    return corpora, stocks

def _read_from_file(filename):
    return tf.map_fn(tf.read_file, filename, dtype=tf.string)

def _embed_text(article, word2vec):
    def embedder(word): 
        def embedder_aux(w):
            if word in word2vec.vocab:
                out = word2vec.word_vec(w.decode("utf-8"))
            else:
                out = word2vec.word_vec(random.choice(word2vec.index2word))
            return out
        return tf.py_func(embedder_aux, [word], tf.float32)
    article = tf.strings.regex_replace(article, tf.constant("[^0-9A-Za-z]"), tf.constant(" "))
    words = tf.string_split([article]).values
    vectors = tf.map_fn(embedder, words, dtype=tf.float32)
    vector = tf.math.reduce_mean(vectors, axis=0)
    return vector

def _prepare_corpus_dataset(corpora, window_size, word2vec, verbose=False):
    dataset = tf.data.Dataset.from_generator(lambda: iter(corpora), output_types=tf.string)
    dataset = dataset.map(_read_from_file)
    if verbose: print_dataset(dataset)
    dataset = dataset.map(lambda x: tf.map_fn(lambda y: _embed_text(y, word2vec), x, dtype=tf.float32))
    if verbose: print_dataset(dataset)
    dataset = dataset.map(tf.transpose)
    if verbose: print_dataset(dataset)
    dataset = dataset.window(window_size, shift=1)
    dataset = dataset.flat_map(lambda subdataset: subdataset \
        .padded_batch(window_size, padded_shapes=[None, None]) \
    )
    if verbose: print_dataset(dataset)
    dataset = dataset.take(len(corpora) - window_size)
    if verbose: print_dataset(dataset)
    return dataset

def _prepare_stock_dataset(stocks, thresholds):
    tensors = []
    for stock in stocks:
        if float(stock) < thresholds[0]:
            new = [1, 0, 0]
        elif float(stock) > thresholds[1]:
            new = [0, 0, 1]
        else:
            new = [0, 1, 0]
        tensors.append(tf.convert_to_tensor(new))
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    return dataset

def _merge_datasets(datasets, verbose=False):
    dataset = datasets[0]
    for other_dataset in datasets[1:]:
        dataset = dataset.concatenate(other_dataset)
    if verbose: print_dataset(dataset)
    return dataset

def input_fn(mode, signal_map, word2vec, params, verbose=False):
    corpus_datasets = []
    stock_datasets = []
    for symbol, (corpora, stocks) in signal_map.items():
        corpus_datasets.append(_prepare_corpus_dataset(corpora, params.window_size, word2vec))
        stock_datasets.append(_prepare_stock_dataset(stocks, params.thresholds).skip(params.window_size))
    corpus_dataset = _merge_datasets(corpus_datasets)
    stock_dataset = _merge_datasets(stock_datasets)
    corpus_dataset = corpus_dataset.padded_batch(params.batch_size, padded_shapes=[None,None,None])
    stock_dataset = stock_dataset.batch(params.batch_size)

    dataset = tf.data.Dataset.zip((corpus_dataset, stock_dataset))
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    corpora, stock = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        "corpora": corpora,
        "stock": stock,
        "iterator_init_op": init_op
    }

    return inputs
