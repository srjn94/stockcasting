import random
import tensorflow as tf

def read_dataset(d, evaluate=False):
    tensors = []
    it = d.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(it.initializer)
        try:
            while True: 
                tensor = sess.run(it.get_next())
                if not evaluate:
                    tensor = tf.constant(tensor)
                tensors.append(tensor)
        except tf.errors.OutOfRangeError:
            pass
    return tensors

def print_dataset(d, evaluate=False):
    for tensor in utils.read_dataset(dataset): 
        print(tensor)

