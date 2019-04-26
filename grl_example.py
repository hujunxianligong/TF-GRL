# coding=utf-8
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

tf.enable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import keras
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


mnist = input_data.read_data_sets("tmp")

batch_size = 128
num_classes = 10

@tf.custom_gradient
def grl(x, alpha):
    def grad(dy):
        return -dy * alpha, None
    return x, grad

encoder_model = keras.Sequential([
    keras.layers.Conv2D(30, 5, activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(60, 4, activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),

])

cls_model = keras.Sequential([
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(num_classes)
])


domain_model = keras.Sequential([
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(2)
])

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)



for step in range(10000):
    source_batch_datas, source_batch_labels = mnist.train.next_batch(batch_size)
    source_batch_datas = source_batch_datas.reshape([-1, 28, 28, 1])

    # print(source_batch_datas[0])
    source_batch_datas = tf.Variable(source_batch_datas, trainable=False)


    target_batch_datas, target_batch_labels = mnist.test.next_batch(batch_size)
    target_batch_datas = 1.0 - target_batch_datas.reshape([-1, 28, 28, 1])
    target_batch_datas = tf.Variable(target_batch_datas, trainable=False)


    with tf.GradientTape() as tape:
        source_batch_encoded = encoder_model(source_batch_datas, training=True)
        source_batch_logits = cls_model(source_batch_encoded, training=True)
        cls_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=source_batch_logits,
            labels=tf.one_hot(source_batch_labels, depth=num_classes)
        )

        target_batch_encoded = encoder_model(target_batch_datas)

        # rate = step / 2000
        # if rate > 1.0:
        #     rate = 1.0
        rate = 1.0
        source_domain_logits = domain_model(grl(source_batch_encoded, rate), training=True)
        target_domain_logits = domain_model(grl(target_batch_encoded, rate), training=True)

        source_domain_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=source_domain_logits,
            labels=tf.ones_like(source_domain_logits)
        )

        target_domain_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=target_domain_logits,
            labels=tf.zeros_like(target_domain_logits)
        )

        # print(cls_losses.shape, source_domain_losses.shape, target_domain_losses.shape)

        losses = cls_losses + source_domain_losses + target_domain_losses

    vars = tape.watched_variables()
    grads = tape.gradient(losses, vars)
    optimizer.apply_gradients(zip(grads, vars))



    if step % 100 == 0:
        mean_loss = tf.reduce_mean(losses)

        target_batch_datas, target_batch_labels = mnist.test.next_batch(batch_size)
        target_batch_datas = 1.0 - target_batch_datas.reshape([-1, 28, 28, 1])
        # target_batch_datas = target_batch_datas.reshape([-1, 28, 28, 1])
        target_batch_datas = tf.Variable(target_batch_datas, trainable=False)

        preds = tf.argmax(cls_model(encoder_model(target_batch_datas)), axis=-1).numpy()

        accuracy = accuracy_score(target_batch_labels, preds)
        print(step, mean_loss, accuracy)

    # print(source_encoded.shape)

    # print(batch_datas.shape)






