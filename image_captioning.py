# -*- coding: utf-8 -*-

# If there's error in pycocotools installation, try installing directly from the repo
# !pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import os
import pickle
from time import time
import json
import re
import sklearn
import keras
import skimage.io as io
import seaborn as sns
import keras.applications.imagenet_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
import warnings

warnings.filterwarnings('ignore')
import pylab

pylab.rcParams['figure.figsize'] = (4, 6)
from pycocotools.coco import COCO

# Test if running on GPU
keras.backend.tensorflow_backend._get_available_gpus()

"""Data"""

# Specify data directory and the COCO training file to be used
# Update the following directory with the new working directory found by the previous command
data_dir = "/home/tcai/Documents/nlp/final_project"
data_type = "train2017"
data_zipfile = '%s.zip' % data_type

# # Download annotation, image, and glove
# annotation_zip = tf.keras.utils.get_file('captions.zip', cache_subdir = os.path.abspath('.'),
#                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
#                                          extract = True)
# image_zip = tf.keras.utils.get_file(data_zipfile, cache_subdir = os.path.abspath('.'),
#                                     origin = 'http://images.cocodataset.org/zips/%s' % (data_zipfile),
#                                     extract = True)
# glove6b_zip = tf.keras.utils.get_file('glove.6B.zip', cache_subdir = os.path.abspath('./glove6b'),
#                                       origin = 'http://nlp.stanford.edu/data/glove.6B.zip',
#                                       extract = True)

# Update file directory objects
annotation_file = data_dir + '/annotations/captions_%s.json' % data_type
image_dir = data_dir + '/%s/' % data_type
coco_caps = COCO(annotation_file)

# Obtain categories
annFile = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
coco = COCO(annFile)

# Specify total number of samples to be used for training and testing.
total_examples = 100000
train_examples = int(0.8 * total_examples)

# Subset training and testing images
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

all_img_path_vector = {}
all_img_name_vector = list(set([x['image_id'] for x in annotations['annotations']]))
print("Choosing %s training images and %s validation images from a total of %s images"
      % (train_examples, total_examples - train_examples, len(all_img_name_vector)))

for img_id in all_img_name_vector:
    img = coco.loadImgs(img_id)[0]
    image_file_path = '%s/%s/%s' % (data_dir, data_type, img['file_name'])
    all_img_path_vector[img_id] = image_file_path

# Shuffle and obtain subset
all_img_name_vector = sklearn.utils.shuffle(all_img_name_vector, random_state = 0)
all_img_name_vector = all_img_name_vector[:total_examples]

# Obtain train and test set
train_img_name_vector = all_img_name_vector[:train_examples]  # train
test_img_name_vector = all_img_name_vector[train_examples:]  # test

"""Image Embeddings"""
# Create the inception v3 model
# take out the fully connected layers at the end to have it output image embeddings
image_model = keras.applications.InceptionV3(weights = 'imagenet')
feature_model = keras.models.Model(image_model.input, image_model.layers[-2].output)

# Encoding function for feature extraction
def encode(image_path):
    # Preprocess images
    img = tf.keras.preprocessing.image.load_img(image_path, target_size = (299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = keras.applications.inception_v3.preprocess_input(img_array)

    # Produce image embeddings
    fea_vec = feature_model.predict(img_array)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return (fea_vec)

# # Encode all the train images. Run once and store features in a pickle file
# start = time()
# train_features = {}
# for img_id in all_img_name_vector:
#     train_features[img_id] = encode(all_img_path_vector[img_id])

# # Took around 2100 seconds.
# print("Time taken in seconds =", time() - start)
#
# # Pickle the features
# with open(data_dir + "/encoded_train_images.pkl", "wb") as encoded_pickle:
#     pickle.dump(train_features, encoded_pickle)

# Load image features from the pickle file
train_features = pickle.load(open(data_dir + "/encoded_train_images.pkl", "rb"))
print('%d photos in total are encoded. These include both training and testing image set.' % len(train_features))

# Create the train image data set
captions = []
images = []
images_features = []

for img_id in train_img_name_vector:
    img_path = all_img_path_vector[img_id]
    img_feature = train_features[img_id]
    img_captions = coco_caps.loadAnns(coco_caps.getAnnIds(img_id))

    for caption in [x['caption'] for x in img_captions]:
        captions.append('start_sentence ' + caption + ' end_sentence')
        images.append(img_path)
        images_features.append(img_feature)

    captions, images, images_features = sklearn.utils.shuffle(captions, images, images_features, random_state = 0)

# Create the validation image data set
test_captions = []
test_images = []
test_images_features = []

for img_id in test_img_name_vector:
    img_path = all_img_path_vector[img_id]
    img_feature = train_features[img_id]
    img_captions = coco_caps.loadAnns(coco_caps.getAnnIds(img_id))

    for caption in [x['caption'] for x in img_captions]:
        test_captions.append('start_sentence ' + caption + ' end_sentence')
        test_images.append(img_path)
        test_images_features.append(img_feature)

    test_captions, test_images, test_images_features = sklearn.utils.shuffle(
        test_captions, test_images, test_images_features, random_state = 0)

print('Training: %s distinct images, %s captions,' %
      (len(list(set(images))), len(captions)))
print('Validation: %s distinct images, %s captions.' %
      (len(list(set(test_images))), len(test_captions)))

"""Word Embeddings"""

# Find the maximum length of any caption in our dataset


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


number_of_words = 6000

# Choose the top words from the vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words = number_of_words,
    oov_token = "<unk>",
    filters = '!"#$%&()*+.,-/:;=?@[\]^`{|}~ ')
captions = [caption.lower() for caption in captions]
tokenizer.fit_on_texts(captions)

# Index the padding values
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Understand the distribution of sentence lengths
sentence_lengths = [len(caption) for caption in captions]
plt.title('Distribution of Caption Lengths')
sns.distplot(sentence_lengths)

# Calculates sentence and vocab lengths
# Manually set the maximum sentence length after observing the distribution
max_length = 100
vocab_size = len(tokenizer.index_word)
print("Original max sentence length is %s; I set it to %s." %
      (calc_max_length(captions), max_length))
print("The vocabulary size is: %s" % (vocab_size))

embedding_dim = 300
embeddings_index = {}

with open(data_dir + '/glove6b/glove.6B.%sd.txt' % (embedding_dim),
          encoding = "utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Get dense vector
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in (tokenizer.word_index).items():
    # All 0 is words not found in the embedding index
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape

"""Model"""

# Extract features
inputs1 = tf.keras.Input(shape = (2048,))
fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
fe2 = tf.keras.layers.Dense(256, activation = 'relu')(fe1)
# Sequence model
inputs2 = tf.keras.Input(shape = (max_length,))
se1 = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                mask_zero = True)(inputs2)
se2 = tf.keras.layers.Dropout(0.5)(se1)
se3 = tf.keras.layers.LSTM(256)(se2)
# Decoder model
decoder1 = tf.keras.layers.Add()([fe2, se3])
decoder2 = tf.keras.layers.Dense(256, activation = 'relu')(decoder1)
outputs = tf.keras.layers.Dense(vocab_size, activation = 'softmax')(decoder2)
# Final model
model = tf.keras.models.Model(inputs = [inputs1, inputs2], outputs = outputs)

# Add the embedding matrix and set the layer to be not trainable
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.summary()

# Generate sentence to break sentence into in and out segments for word-by-word prediction
def generate_sequences(tokenizer, max_length, caption, image_feature):
    Ximages, XSeq, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index)
    seq = tokenizer.texts_to_sequences([caption])[0]
    # Split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # Select substrings
        in_seq, out_seq = seq[:i], seq[i]
        # Pad input sequence
        in_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [in_seq], maxlen = max_length, padding = 'post')[0]
        # Encode output sequence
        out_seq = tf.keras.utils.to_categorical([out_seq],
                                                num_classes = vocab_size)[0]

        image_feature = np.squeeze(image_feature)
        Ximages.append(image_feature)
        XSeq.append(in_seq)
        y.append(out_seq)

    # Connect sentence sequence with images and the output sequence
    Ximages, XSeq, y = np.array(Ximages), np.array(XSeq), np.array(y)
    return [Ximages, XSeq, y]


# Python data generator object to loop through all images
def data_generator(tokenizer, max_length, captions, images_features, batch_size):
    n = 0
    while True:
        for i in range(len(captions)):
            in_img_vector = []
            in_seq_vector = []
            out_word_vector = []
            # Load image feature
            image_feature = images_features[i]
            # Generate word sequence
            caption = captions[i]
            in_img, in_seq, out_word = generate_sequences(
                tokenizer, max_length, caption, image_feature)
            in_img_vector.append(in_img)
            in_seq_vector.append(in_seq)
            out_word_vector.append(out_word)
            n += 1
            # When batch size is reached, yield the output
            if n == batch_size:
                n = 0
                yield [in_img, in_seq], out_word


# Predict caption
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'start_sentence'
    for i in range(max_length):
        # Use input and image to start predict the rest of the caption
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence], maxlen = max_length, padding = 'post')
        photo = photo.reshape(2048, 1).T

        # Predict the next word based on sequence and image
        yhat = model.predict([photo, sequence], verbose = 0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]

        # End prediction when no word is predicted or if ending word is seen
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end_sentence':
            break

    # Remove beginning and ending signal words when output
    in_text = re.sub(r'(start|end)_sentence', '', in_text).strip()
    return in_text


# Store default weights in the model directory
model_directory = './model'

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model.save_weights(model_directory + '/model.h5')

# Reset model with default weights before training
model.load_weights(model_directory + '/model.h5')

# Specify final model parameters
# After trying out different batch sizes, chose the one that provided the most reasonable results
epochs = 50
batch_size = 500
steps = len(captions) // batch_size
epoch_idx = 0

# Create dictionary to store loss values. Need this because using fit generator
loss = {}

# Use fit generator due to limited memory.
while epoch_idx <= epochs:
    # Use data generator to input data
    generator = data_generator(tokenizer, max_length, captions,
                               images_features, batch_size)
    # Fit generator is used due to memory limitation
    history = model.fit_generator(generator, steps_per_epoch = steps, verbose = 1)
    loss[epoch_idx] = history.history['loss']
    model.save_weights(model_directory + '/model_epoch%s_bs%s.h5' %
                       (epoch_idx, batch_size))

    epoch_idx += 1

    # Keep track of the number of distinct captions
    distinct_desc = []
    temp_desc = ''
    for j in range(20):
        img_desc = generate_desc(model, tokenizer, images_features[j], max_length)
        distinct_desc.append(img_desc)
        temp_desc += '' + img_desc

    print("Model %s generated %s distinct captions with %s distinct words." %
          (epoch_idx, len(list(set(distinct_desc))), len(list(set(temp_desc.split(' '))))))

# Return the loss plot
plt.title('Model Training Loss')

loss_lists = sorted(loss.items())
x, y = zip(*loss_lists)
plt.plot(x, y)

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

# todo Add image transformation to improve learning
# todo BLEU calculation here might be problematic
# todo Calculate overall BLEU score
# Train images
print(
    "Sample prediction of the model with a batch size of %s and %s epochs.\n" %
    (batch_size, epochs))

# Load the latest model weight
model.load_weights(model_directory + '/model_epoch%s_bs%s.h5' %
                   (epochs, batch_size))

# Output prediction for the first five images in the validation data. 
# Note that the data set is already shuffled.
np.random.seed(101)
for j in np.random.choice(range(len(test_captions)), 5):
    # Compare captions
    img_desc = generate_desc(model, tokenizer, test_images_features[j],
                             max_length)
    actual_caption = test_captions[j]
    actual_caption = re.sub(r'(start|end)_sentence', '', actual_caption)

    print("Predicted Caption: %s \nActual Caption: %s" %
          (img_desc, actual_caption))
    result_bleu = nltk.translate.bleu_score.sentence_bleu(actual_caption,
                                                          img_desc)
    print("Resulting BLEU-4 score is %s" % (result_bleu))
    # Show image
    I = io.imread(test_images[j])
    plt.imshow(I)
    plt.axis('off')
    plt.show()

"""## Tuning
"""

# Trying out different batch sizes
for batch_size in [500, 1000, 2000, 3000]:
    loss = {}
    model.load_weights(model_directory + '/model.h5')
    steps = len(captions) // batch_size
    print(batch_size)
    epoch_idx = 0
    while epoch_idx <= epochs:
        generator = data_generator(tokenizer, max_length, captions,
                                   images_features, batch_size)
        history = model.fit_generator(generator,
                                      steps_per_epoch = steps,
                                      verbose = 1)
        loss[epoch_idx] = history.history['loss']

        model.save_weights(model_directory + '/model_epoch' + str(epoch_idx) +
                           "_bs" + str(batch_size) + '.h5')
        epoch_idx += 1

        distinct_desc = []
        temp_desc = ''
        for j in range(20):
            img_desc = generate_desc(model, tokenizer, images_features[j],
                                     max_length)
            distinct_desc.append(img_desc)
            temp_desc += '' + img_desc

        print("%s distinct captions, %s distinct words." % (len(
            list(set(distinct_desc))), len(list(set(temp_desc.split(' '))))))

    # Plot model loss
    plt.title('Model Training Loss')

    loss_lists = sorted(
        loss.items())  # sorted by key, return a list of tuples
    x, y = zip(*loss_lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

"""### Test on Training Data
"""

np.random.seed(17)

rand_train_image = np.random.choice(range(len(captions)))

I = io.imread(images[rand_train_image])
plt.imshow(I)
plt.axis('off')
plt.show()

# Obtain actual and predicted images and their captions
actual_caption = captions[rand_train_image]
actual_caption = re.sub(r'(start|end)_sentence', '', actual_caption).strip()
print("The actual caption is:\n%s \n\nThe predicted captions are:" %
      (actual_caption))

for bs in [500, 1000, 2000, 3000]:
    model.load_weights(model_directory + '/model_epoch%s_bs%s.h5' %
                       (epochs - 1, bs))
    # Compare captions
    img_desc = generate_desc(model, tokenizer,
                             images_features[rand_train_image], max_length)

    actual_caption = re.sub(r'(start|end)_sentence', '', actual_caption)
    print("Batch size = %s: %s" % (bs, img_desc))

    result_bleu = nltk.translate.bleu_score.sentence_bleu(
        actual_caption, img_desc)
    print("Resulting BLEU-4 score is %s\n" % (result_bleu))

"""### Test on Validation Data"""

np.random.seed(511)

rand_test_image = np.random.choice(range(len(test_captions)))

I = io.imread(images[rand_test_image])
plt.imshow(I)
plt.axis('off')
plt.show()

# Obtain actual and predicted images and their captions
actual_caption = captions[rand_test_image]
actual_caption = re.sub(r'(start|end)_sentence', '', actual_caption).strip()
print("The actual caption is:\n%s \n\nThe predicted captions are:" %
      (actual_caption))

for bs in [500, 1000, 2000, 3000]:
    model.load_weights(model_directory + '/model_epoch%s_bs%s.h5' %
                       (epochs - 1, bs))
    # Compare captions
    img_desc = generate_desc(model, tokenizer,
                             images_features[rand_test_image], max_length)

    actual_caption = re.sub(r'(start|end)_sentence', '', actual_caption)
    print("Batch size = %s: %s" % (bs, img_desc))

    result_bleu = nltk.translate.bleu_score.sentence_bleu(
        actual_caption, img_desc)
    print("Resulting BLEU-4 score is %s\n" % (result_bleu))

"""Discussion"""
# Understand vocab distribution to showcase the model bias
# Note that identifying stop words and lemmatization takes a while.
word_list = []
for caption in captions:
    seq = caption.split(" ")
    word_list += seq

# Remove idiosyncracies in word usage
lemmatizer = WordNetLemmatizer()
word_list = [
    lemmatizer.lemmatize(word.lower()) for word in word_list
    if word and word.lower() not in nltk.corpus.stopwords.words('english')
       and "_" not in word
]

# Calculate frequency distribution
word_list_dist = nltk.FreqDist(word_list)

# Run this line to output most frequently used vocabularies in the data set
limit = 30
word_dict = dict()
for word, frequency in word_list_dist.most_common(limit):
    print(u'{}\t\t{}'.format(word, frequency))
    word_dict[word] = frequency


def show_img_example(img_idx):
    # Show images given index
    I = io.imread(images[img_idx])
    plt.imshow(I)
    plt.axis('off')
    plt.show()

    # Create actual and predicted caption
    actual_caption = captions[img_idx]
    actual_caption = re.sub(r'(start|end)_sentence', '', actual_caption).strip()

    model.load_weights(model_directory + '/model_epoch50_bs500.h5')
    img_desc = generate_desc(model, tokenizer, images_features[img_idx],
                             max_length)

    print("The actual caption is:\n%s \n\nThe predicted captions is:\n%s" %
          (actual_caption, img_desc))

    result_bleu = nltk.translate.bleu_score.sentence_bleu(
        actual_caption, img_desc)
    print("\nResulting BLEU-4 score is %s\n" % (result_bleu))


show_img_example(9836)

img_idx = np.random.choice(range(len(test_captions)))
show_img_example(img_idx)
print(img_idx)