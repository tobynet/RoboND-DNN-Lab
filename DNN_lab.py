
# coding: utf-8

# <h1 align="center">TensorFlow Deep Neural Network Lab</h1>

# <img src="image/notmnist.png">
# In this lab, you'll use all the tools you learned from the *Deep Neural Networks* lesson to label images of English letters! The data you are using, <a href="http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html">notMNIST</a>, consists of images of a letter from A to J in differents font.
# 
# The above images are a few examples of the data you'll be training on. After training the network, you will compare your prediction model against test data. While there is no predefined goal for this lab, we would like you to experiment and discuss with fellow students on what can improve such models to achieve the highest possible accuracy values.

# To start this lab, you first need to import all the necessary modules. Run the code below. If it runs successfully, it will print "`All modules imported`".

# In[1]:


import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

print('All modules imported.')
dataset_directory = '../datasets/'


# The notMNIST dataset is too large for many computers to handle.  It contains 500,000 images for just training.  You'll be using a subset of this data, 15,000 images for each label (A-J).

# In[2]:




def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    file = os.path.join(dataset_directory, file)
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

# Download the training and test dataset.
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

# Make sure the files aren't corrupted
assert hashlib.md5(open(os.path.join(dataset_directory, 'notMNIST_train.zip'), 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
assert hashlib.md5(open(os.path.join(dataset_directory, 'notMNIST_test.zip'), 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

# Wait until you see that all files have been downloaded.
print('All files downloaded.')


# In[3]:


def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []
    
    file = os.path.join(dataset_directory, file)
    
    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Get the features and labels from the zip files
train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

# Limit the amount of data to work with
size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=size_limit)

# Set flags for feature engineering.  This will prevent you from skipping an important step.
is_features_normal = False
is_labels_encod = False

# Wait until you see that all features and labels have been uncompressed.
print('All features and labels uncompressed.')


# <img src="image/mean_variance.png" style="height: 75%;width: 75%; position: relative; right: 5%">
# ## Problem 1
# The first problem involves normalizing the features for your training and test data.
# 
# Implement Min-Max scaling in the `normalize()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.
# 
# Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.
# 
# Min-Max Scaling:
# $
# X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
# $

# In[9]:


# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # DONE: Implement Min-Max scaling for grayscale image data
    a = 0.1
    b = 0.9
    x = image_data
    min_of_x = np.min(x)
    max_of_x = np.max(x)
    return a + (x-min_of_x) * (b-a) / (max_of_x-min_of_x)    


### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
    [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9],
    decimal=3)
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
    [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9])

if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

print('Tests Passed!')


# In[10]:


if not is_labels_encod:
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

print('Labels One-Hot Encoded')


# In[11]:


assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'

# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')


# In[12]:


# Save the data for easy access
import gzip # サイズが 500MB 近いので圧縮したい

pickle_file = os.path.join(dataset_directory, 'notMNIST.pickle.gz')
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with gzip.open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')


# # Checkpoint
# All your progress is now saved to the pickle file.  If you need to leave and comeback to this lab, you no longer have to start from the beginning.  Just run the code block below and it will load all the data and modules required to proceed.

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Load the modules
import pickle
import gzip # 圧縮されたキャッシュ用
import math
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
dataset_directory = '../datasets/'

pickle_file = os.path.join(dataset_directory, 'notMNIST.pickle.gz')
with gzip.open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')


# <img src="image/weight_biases.png" style="height: 60%;width: 60%; position: relative; right: 10%">
# ## Problem 2
# For the neural network to train on your data, you need the following <a href="https://www.tensorflow.org/resources/dims_types.html#data-types">float32</a> tensors:
#  - `features`
#   - Placeholder tensor for feature data (`train_features`/`valid_features`/`test_features`)
#  - `labels`
#   - Placeholder tensor for label data (`train_labels`/`valid_labels`/`test_labels`)
#  - `keep_prob`
#   - Placeholder tensor for dropout's keep probability value
#  - `weights`
#   - List of Variable Tensors with random numbers from a truncated normal distribution for each list index.
#     - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#truncated_normal">`tf.truncated_normal()` documentation</a> for help.
#  - `biases`
#   - List of Variable Tensors with all zeros for each list index.
#     - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#zeros"> `tf.zeros()` documentation</a> for help.

# In[5]:


tf.reset_default_graph()

features_count = 784
labels_count = 10

# TODO: Set the hidden layer width. You can try different widths for different layers and experiment.
hidden_layer_width = 50

# TODO: Set the features, labels, and keep_prob tensors
# 入力用 placeholder
features = tf.placeholder(tf.float32, [None, features_count])
# 出力用 placeholder
labels = tf.placeholder(tf.float32, [None, labels_count])
# dropout用
keep_prob = tf.placeholder(tf.float32)


# TODO: Set the list of weights and biases tensors based on number of layers
weights = [
    tf.Variable(tf.truncated_normal([features_count, hidden_layer_width], stddev=0.01)),
    tf.Variable(tf.truncated_normal([hidden_layer_width, hidden_layer_width], stddev=0.01)),
#    tf.Variable(tf.truncated_normal([hidden_layer_width, hidden_layer_width], stddev=0.01)),
    tf.Variable(tf.truncated_normal([hidden_layer_width, labels_count], stddev=0.01))
]

biases = [
    tf.Variable(tf.zeros([hidden_layer_width])),
    tf.Variable(tf.zeros([hidden_layer_width])),
#    tf.Variable(tf.zeros([hidden_layer_width])),
    tf.Variable(tf.zeros([labels_count]))
]



### DON'T MODIFY ANYTHING BELOW ###
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert all(isinstance(weight, Variable) for weight in weights), 'weights must be a TensorFlow variable'
assert all(isinstance(bias, Variable) for bias in biases), 'biases must be a TensorFlow variable'

assert features.shape == None or (    features.shape.dims[0].value is None and    features.shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels.shape  == None or (    labels.shape.dims[0].value is None and    labels.shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

print("done")


# 
# ## Problem 3
# This problem would help you implement the hidden and output layers of your model. As it was covered in the classroom, you will need the following:
# 
# - [tf.add](https://www.tensorflow.org/api_docs/python/tf/add) and [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul) to create your hidden and output(logits) layers.
# - [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu) for your ReLU activation function.
# - [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) for your dropout layer.

# In[6]:


if 'hidden_layer_1' in globals(): del hidden_layer_1; print('deleted')
if 'hidden_layer_2' in globals(): del hidden_layer_2; print('deleted')
if 'hidden_layer_3' in globals(): del hidden_layer_3; print('deleted')
if 'logits' in globals(): del logits; print('deleted')


# In[7]:


# TODO: Hidden Layers with ReLU Activation and dropouts. "features" would be the input to the first layer.
hidden_layer_num = len(weights) - 1
hidden_layers = []

layer = features
for i in range(hidden_layer_num):
    layer = layer @ weights[i] + biases[i]
    layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    hidden_layers.append(layer)
    
    print('  make hidden_layer {}: {}'.format(i, layer))
    
logits = hidden_layers[-1] @ weights[-1] + biases[-1]

print('hidden_layer_num:', hidden_layer_num)
print('hidden_layers: ', len(hidden_layers))

# hidden_layer_1 = features @ weights[0] + biases[0]
# hidden_layer_1 = tf.nn.relu(hidden_layer_1)
# hidden_layer_1 = tf.nn.dropout(hidden_layer_1, keep_prob=keep_prob)

# hidden_layer_2 = hidden_layer_1 @ weights[1] + biases[1]
# hidden_layer_2 = tf.nn.relu(hidden_layer_2)
# hidden_layer_2 = tf.nn.dropout(hidden_layer_2, keep_prob=keep_prob)

# # TODO: Output layer
# logits = hidden_layer_2 @ weights[2] + biases[2]


# In[8]:


### DON'T MODIFY ANYTHING BELOW ###

prediction = tf.nn.softmax(logits)

# Training loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# Create an operation that initializes all variables
init = tf.global_variables_initializer()

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')


# <img src="image/learn_rate_tune.png" style="height: 60%;width: 60%">
# ## Problem 4
# In the previous lab for a single Neural Network, you attempted several different configurations for the hyperparameters given below. Try to first use the same parameters as the previous lab, and then adjust and finetune those values based on your new model if required. 
# 
# You have another hyperparameter to tune now, however. Set the value for keep_probability and observe how it affects your results.

# In[9]:


get_ipython().run_cell_magic('time', '', "# TODO: Find the best parameters for each configuration\nepochs = 10\nbatch_size = 50\nlearning_rate = 0.01\nkeep_probability = 1.0\n\n\n### DON'T MODIFY ANYTHING BELOW ###\n# Gradient Descent\noptimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    \n\n# The accuracy measured against the validation set\nvalidation_accuracy = 0.0\n\n# Measurements use for graphing loss and accuracy\nlog_batch_step = 50\nbatches = []\nloss_batch = []\ntrain_acc_batch = []\nvalid_acc_batch = []\n\nwith tf.Session() as session:\n    session.run(init)\n    batch_count = int(math.ceil(len(train_features)/batch_size))\n\n    for epoch_i in range(epochs):\n        \n        # Progress bar\n        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')\n        \n        # The training cycle\n        for batch_i in batches_pbar:\n            # Get a batch of training features and labels\n            batch_start = batch_i*batch_size\n            batch_features = train_features[batch_start:batch_start + batch_size]\n            batch_labels = train_labels[batch_start:batch_start + batch_size]\n\n            # Run optimizer and get loss\n            _, l = session.run(\n                [optimizer, loss],\n                feed_dict={features: batch_features, labels: batch_labels, keep_prob: keep_probability})\n\n            # Log every 50 batches\n            if not batch_i % log_batch_step:\n                # Calculate Training and Validation accuracy\n                training_accuracy = session.run(accuracy, feed_dict={features: train_features, \n                                                                     labels: train_labels, keep_prob: keep_probability})\n                validation_accuracy = session.run(accuracy, feed_dict={features: valid_features, \n                                                                     labels: valid_labels, keep_prob: 1.0})\n                # Log batches\n                previous_batch = batches[-1] if batches else 0\n                batches.append(log_batch_step + previous_batch)\n                loss_batch.append(l)\n                train_acc_batch.append(training_accuracy)\n                valid_acc_batch.append(validation_accuracy)\n\n        # Check accuracy against Validation data\n        validation_accuracy = session.run(accuracy, feed_dict={features: valid_features, \n                                                                     labels: valid_labels, keep_prob: 1.0})\n        print('  Validation accuracy at {}'.format(validation_accuracy))\n\n\nloss_plot = plt.subplot(211)\nloss_plot.set_title('Loss')\nloss_plot.plot(batches, loss_batch, 'g')\nloss_plot.set_xlim([batches[0], batches[-1]])\nacc_plot = plt.subplot(212)\nacc_plot.set_title('Accuracy')\nacc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')\nacc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')\nacc_plot.set_ylim([0, 1.0])\nacc_plot.set_xlim([batches[0], batches[-1]])\nacc_plot.legend(loc=4)\nplt.tight_layout()\nplt.show()\n\nprint('Validation accuracy at {}'.format(validation_accuracy))")


# ## Test
# Set the epochs, batch_size, and learning_rate with the best learning parameters you discovered in problem 4.  You're going to test your model against your hold out dataset/testing data.  This will give you a good indicator of how well the model will do in the real world.

# In[10]:


get_ipython().run_cell_magic('time', '', "# TODO: Set the epochs, batch_size, and learning_rate with the best parameters from problem 4\n#epochs = None\n#batch_size = None \n#learning_rate = None\nepochs = 10\nbatch_size = 50\nlearning_rate = 0.01\n\n\n### DON'T MODIFY ANYTHING BELOW ###\n# The accuracy measured against the test set\ntest_accuracy = 0.0\n\nwith tf.Session() as session:\n    \n    session.run(init)\n    batch_count = int(math.ceil(len(train_features)/batch_size))\n\n    for epoch_i in range(epochs):\n        \n        # Progress bar\n        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')\n        \n        # The training cycle\n        for batch_i in batches_pbar:\n            # Get a batch of training features and labels\n            batch_start = batch_i*batch_size\n            batch_features = train_features[batch_start:batch_start + batch_size]\n            batch_labels = train_labels[batch_start:batch_start + batch_size]\n\n            # Run optimizer\n            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels, keep_prob: 1.0})\n\n        # Check accuracy against Test data\n        test_accuracy = session.run(accuracy, feed_dict={features: test_features, \n                                                                     labels: test_labels, keep_prob: 1.0})\n\nprint('Nice Job! Test Accuracy is {}'.format(test_accuracy))")


# ## 試しにモデルを保存
# 
# ダメ。 学習セッション直後でなくてはいけないのかも

# In[52]:


save_file = './saved-models/train_model.ckpt'

saver = tf.train.Saver()

with tf.Session() as session:
    # Save the model
    saver.save(session, save_file)
    print('Trained Model Saved.')

