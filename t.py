#Test Dataset.shuffle and Dataset.repeat

import tensorflow as tf

n_epochs = 2
batch_size = 3

data = tf.data.Dataset.range(12)

data = data.shuffle(12)
data = data.repeat(n_epochs)

data = data.batch(batch_size)
next_batch = data.make_one_shot_iterator().get_next()

sess = tf.Session()

print(sess.run(next_batch))
print(sess.run(next_batch))
print(sess.run(next_batch))

