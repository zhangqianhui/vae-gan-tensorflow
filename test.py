import os
import tensorflow as tf
import numpy as np

a = [[1,2],[3,4],[4,6]]
b = [[3,5],[6,7],[8,9]]

a = np.array(a, dtype=float)

result = tf.reduce_sum(a)
result2 = tf.reduce_mean(a)
result3 = tf.reduce_mean(result)


with tf.Session() as sess:

    print sess.run(result)
    print sess.run(result2)

