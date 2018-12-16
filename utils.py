import random
import tensorflow as tf

def read_dataset(d, n=1, start=0, mode="tensors"):
    tensors = []
    it = d.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(it.initializer)
        for _ in range(start, start + n):
            if mode == "arrays":
                tensors.append(sess.run(it.get_next()))
            else:
                tensors.append(tf.constant(sess.run(it.get_next())))
    return tensors

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
