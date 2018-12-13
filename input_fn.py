import random
import tensorflow as tf

def strip_punctuation(text):
    text = tf.strings.regex_replace(text, tf.constant("[^0-9A-Za-z]"), tf.constant(" "))
    return text

def embed_text(article, model):
    def embedder(word): 
        def embedder_aux(w):
            if word in model.vocab:
                return model.word_vec(w.decode("utf-8"))
            return model.word_vec(random.choice(model.index2word))
        return tf.py_func(embedder_aux, [word], tf.float32)
    article = strip_punctuation(article)
    words = tf.string_split([article]).values
    vectors = tf.map_fn(embedder, words, dtype=tf.float32)
    vector = tf.math.reduce_mean(vectors, axis=0)
    return article, words, vector


