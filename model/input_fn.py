import os
import random
import tensorflow as tf
from datetime import datetime

def _strip_punctuation(text):
    text = tf.strings.regex_replace(text, tf.constant("[^0-9A-Za-z]"), tf.constant(" "))
    return text

def _embed_text(article, word2vec):
    def embedder(word): 
        def embedder_aux(w):
            if word in word2vec.vocab:
                return word2vec.word_vec(w.decode("utf-8"))
            return word2vec.word_vec(random.choice(word2vec.index2word))
        return tf.py_func(embedder_aux, [word], tf.float32)
    article = _strip_punctuation(article)
    words = tf.string_split([article]).values
    vectors = tf.map_fn(embedder, words, dtype=tf.float32)
    vector = tf.math.reduce_mean(vectors, axis=0)
    return vector

def load_signal(path, corpus_path, symbol):
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
    return corpora, stocks

def _filter_overlapping_values(x, window_size):
    s1 = tf.slice(x[0], [window_size//2, 0], [-1, -1])
    s2 = tf.slice(x[1], [0, 0], [window_size//2, -1])
    return tf.concat((s1, s2), axis=0)

def _prepare_corpus_dataset(corpora, window_size, word2vec):
    def generator():
        for corpus in corpora:
            yield corpus
    return tf.data.Dataset.from_generator(generator, output_types=tf.string) \
        .map(lambda x: tf.map_fn(tf.read_file, x, dtype=tf.string)) \
        .map(lambda x: tf.map_fn(lambda y: _embed_text(y, word2vec), x, dtype=tf.float32)) \

def _prepare_stock_dataset(stocks, thresholds):
    dataset = []
    for stock in stocks:
        if float(stock) < thresholds[0]:
            new = [1, 0, 0]
        elif float(stock) > thresholds[1]:
            new = [0, 0, 1]
        else:
            new = [0, 1, 0]
        dataset.append(tf.convert_to_tensor(new))
    return tf.data.Dataset.from_tensor_slices(dataset)

def input_fn(mode, signal_map, word2vec, params):
    datasets = []
    for symbol, (corpora, stocks) in signal_map.items():
        corpus_dataset = _prepare_corpus_dataset(corpora, params.window_size, word2vec)
        stock_dataset = _prepare_stock_dataset(stocks, params.thresholds).skip(params.window_size)
        datasets.append(tf.data.Dataset.zip((corpus_dataset, stock_dataset)))
    dataset = datasets[0]
    for other_dataset in datasets[1:]:
        dataset = dataset.batch(10).concatenate(other_dataset)
    dataset = dataset.prefetch(1)
        
    iterator = dataset.make_initializable_iterator()
    corpora, stock = iterator.get_next()
    print(corpora)
    print(stock)
    init_op = iterator.initializer

    inputs = {
        "corpora": corpora,
        "stock": stock,
        "iterator_init_op": init_op
    }

    return inputs
