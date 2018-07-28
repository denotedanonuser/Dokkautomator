"""
Data loading heavily inspired from:
https://github.com/yeephycho/tensorflow_input_image_by_tfrecord

Network Architecture taken from:
https://github.com/rahulbhalley/AlexNet-TensorFlow

"""

import tensorflow as tf
import os


DATA_DIR = "./output/"
TRAINING_SET_SIZE = 249
BATCH_SIZE = 10
IMAGE_SIZE = 227
n_classes = 15


# Image Object Based on Protobuf information
class ImageObject:
    def __init__(self):
        self.image = tf.Variable([], dtype=tf.string)
        self.height = tf.Variable([], dtype=tf.int64)
        self.width = tf.Variable([], dtype=tf.int64)
        self.filename = tf.Variable([], dtype=tf.string)
        self.label = tf.Variable([], dtype=tf.int32)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    image_object = ImageObject()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object


def image_input(if_random=True, if_training=True):
    if if_training:
        file_names = [os.path.join(DATA_DIR, "train-0{}-of-04.tfrecord".format(i)) for i in range(0, 4)]
    else:
        file_names = [os.path.join(DATA_DIR, "validation-0{}-of-04.tfrecord".format(i)) for i in range(0, 4)]

    for f in file_names:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(file_names)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
    label = image_object.label
    filename = image_object.filename

    if if_random:
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size=BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue=min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size=BATCH_SIZE,
            num_threads=1)
        return image_batch, label_batch, filename_batch


# Weight parameters as devised in the original research paper
weights = {
    "wc1": tf.Variable(tf.truncated_normal([11, 11, 3, 96],     stddev=0.01), name="wc1"),
    "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name="wc2"),
    "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name="wc3"),
    "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name="wc4"),
    "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name="wc5"),
    "wf1": tf.Variable(tf.truncated_normal([6*6*256, 4096],   stddev=0.01), name="wf1"),
    "wf2": tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name="wf2"),
    "wf3": tf.Variable(tf.truncated_normal([4096, n_classes],   stddev=0.01), name="wf3")
}

# Bias parameters as devised in the original research paper
biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[96]), name="bc1"),
    "bc2": tf.Variable(tf.constant(1.0, shape=[256]), name="bc2"),
    "bc3": tf.Variable(tf.constant(0.0, shape=[384]), name="bc3"),
    "bc4": tf.Variable(tf.constant(1.0, shape=[384]), name="bc4"),
    "bc5": tf.Variable(tf.constant(1.0, shape=[256]), name="bc5"),
    "bf1": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf1"),
    "bf2": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf2"),
    "bf3": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bf3")
}

fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)


def alex_net(img, weights, biases):
    """Neural Network based on the AlexNet, taken from: https://github.com/rahulbhalley/AlexNet-TensorFlow
    :param img: image batch
    :param weights: weights to train on
    :param biases: biases to train on
    :return: final output layer
    """

    # reshape the input image vector to 227 x 227 x 3 dimensions
    img = tf.reshape(img, [-1, 227, 227, 3])

    # 1st convolutional layer
    conv1 = tf.nn.conv2d(img, weights["wc1"], strides=[1, 4, 4, 1], padding="SAME", name="conv1")
    conv1 = tf.nn.bias_add(conv1, biases["bc1"])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 2nd convolutional layer
    conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
    conv2 = tf.nn.bias_add(conv2, biases["bc2"])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 3rd convolutional layer
    conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
    conv3 = tf.nn.bias_add(conv3, biases["bc3"])
    conv3 = tf.nn.relu(conv3)

    # 4th convolutional layer
    conv4 = tf.nn.conv2d(conv3, weights["wc4"], strides=[1, 1, 1, 1], padding="SAME", name="conv4")
    conv4 = tf.nn.bias_add(conv4, biases["bc4"])
    conv4 = tf.nn.relu(conv4)

    # 5th convolutional layer
    conv5 = tf.nn.conv2d(conv4, weights["wc5"], strides=[1, 1, 1, 1], padding="SAME", name="conv5")
    conv5 = tf.nn.bias_add(conv5, biases["bc5"])
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # stretching out the 5th convolutional layer into a long n-dimensional tensor
    shape = [-1, weights['wf1'].get_shape().as_list()[0]]
    flatten = tf.reshape(conv5, shape)

    # 1st fully connected layer
    fc1 = fc_layer(flatten, weights["wf1"], biases["bf1"], name="fc1")
    fc1 = tf.nn.tanh(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    # 2nd fully connected layer
    fc2 = fc_layer(fc1, weights["wf2"], biases["bf2"], name="fc2")
    fc2 = tf.nn.tanh(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    # 3rd fully connected layer
    fc3 = fc_layer(fc2, weights["wf3"], biases["bf3"], name="fc3")
    fc3 = tf.nn.softmax(fc3)

    # Return the complete AlexNet model
    return fc3


def train(insert):
    image_batch_out, label_batch_out, filename_batch = image_input(if_random=True, if_training=True)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[10, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (10, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[10, n_classes])
    label_offset = -tf.ones([10], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=n_classes, on_value=1.0, off_value=0.0)

    logits_out = alex_net(image_batch_placeholder, weights, biases)
    loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)

    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

    data = open('output.txt', 'w')
    with tf.Session() as sess:
        print("Starting Training Session")

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        cum_loss = 0
        for i in range(insert):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch,
                                                                                    label_batch_out,
                                                                                    label_batch_one_hot,
                                                                                    filename_batch])

            _, infer_out, loss_out = sess.run([train_step, logits_out, loss],
                                              feed_dict={image_batch_placeholder: image_out,
                                                         label_batch_placeholder: label_batch_one_hot_out})

            cum_loss += loss_out
            print("THIS IS THE LOSS {} AT ITERATION {}".format(loss_out, i))

            if not i % 100 and i != 0:
                print("===================================")
                print(i)
                print("AVERAGE LOSS PER 100 ITERATIONS: {}".format(cum_loss / 100))
                data.write("AT {} LOSS WAS {}\n".format(i, (cum_loss / 100)))
                cum_loss = 0

        data.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()


def evaluate(insert):
    image_batch_out, label_batch_out, filename_batch = image_input(if_random=True, if_training=False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch_out, label_offset)

    logits_out = tf.reshape(alex_net(image_batch_placeholder, weights, biases), [BATCH_SIZE, n_classes])
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension=1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0

        for i in range(insert):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out

            print(i)
            print(image_out.shape)
            print("label_out: ")
            print(filename_out)
            print(label_out)
            print(logits_batch_out)

        print("Accuracy at {}:  ".format(insert))
        print(accuracy_accu / insert)
        accuracy_outer = accuracy_accu / insert

        coord.request_stop()
        coord.join(threads)
        sess.close()

    return accuracy_outer


iteration_count = 10000

acc = evaluate(iteration_count)
file = open("accuracystart.txt", 'w')
file.write("{}".format(acc))
file.close()

train(iteration_count)

acc = evaluate(iteration_count)
file = open("output.txt".format(iteration_count), 'w')
file.write("{}".format(acc))
file.close()
