import keras, os
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras import backend as K
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import numpy as np

from read_config import *
import util as _

class evaluate(keras.callbacks.Callback):
    def __init__(self, w2v, model, eval_x, eval_y):
        # Get all settings from config file
        self.interval = get_int('devise', 'eval_interval')
        self.topn = get_int('devise', 'eval_topn')
        self.model_save_dir = os.path.join(os.getcwd(), get_str('devise', 'model_save_dir'))
        self.model_name = get_str('devise', 'model_name')
        self.w2v = w2v
        self.model = model
        self.eval_x = eval_x
        self.eval_y = eval_y

    def eval_hit(self, eval_type):
        # Look up in word2vec model to determine correct or not
        x, y = self.eval_x[eval_type], self.eval_y[eval_type]
        # An array to record hit number
        topn_hit = np.zeros(self.topn)
        y_pred = self.model.predict(x)
        for yi in range(len(y_pred)):
            # Feed vector from nn to word2vec model, get top n similar vector
            vec = self.w2v.similar_by_vector(y_pred[yi], topn=self.topn)
            for i in range(self.topn):
                if vec[i][0] == y[yi]:
                    # If hit, count number
                    topn_hit[i:] += 1
                    break
        return topn_hit / x.shape[0]

    def save_model(self, epoch):
        # Save model to predefined location
        if not os.path.isdir(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        file_name = self.model_name.format(epoch)
        self.model.save(os.path.join(self.model_save_dir, file_name))
        print('{} saved!'.format(file_name))

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.interval == 0:
            # Evaluate model accuracy
            print('\nEvaluation process start .....')
            # Get hit result and format to a single string
            topn_hit_flat = ', '.join(str(x) for x in self.eval_hit('flat') * 100)
            topn_hit_zeroshot = ', '.join(str(x) for x in self.eval_hit('zeroshot') * 100)
            # Write result to log file and std out
            log_text = 'Epoch: {0},\nTopK flat: {1}(%)\nTopK zeroshot: {2}(%)\n'.format(epoch + 1, topn_hit_flat, topn_hit_zeroshot)
            _.file_logger(log_text)
            print(log_text)
            # Then save model
            self.save_model(epoch + 1)

class DeViSE:
    def __init__(self):
        # Get all settings from config file
        self.batch_size = get_int('devise', 'batch_size')
        self.visual_model_path = os.path.join(os.getcwd(), 
                                            get_str('visual', 'model_save_dir'),
                                            get_str('visual', 'model_name'))
        self.lr = get_float('devise', 'learning_rate')
        self.epochs = get_int('devise', 'epochs')
        self.freeze_layers = get_bool('devise', 'freeze_visual_layers')
        self.num_classes = get_int('visual', 'num_classes')
        self.sample_or_not = get_bool('devise', 'eval_sample')
        self.word_vectors, self.word_dim = self.load_w2v_model()
        self.label_vec, self.label_text = self.build_label_to_vt()
        self.label_vec_tensor = tf.constant(self.label_vec[:self.num_classes], dtype='float32')
        self.load_data()
        self.model = self.build_model()
        print('Start training DeViSE model.')
        self.train_ops()

    def load_w2v_model(self):
        # Load pre-trained word2vec model
        model_loc = os.path.join(os.getcwd(),get_str('devise', 'w2v_model_name'))
        word_vectors = KeyedVectors.load_word2vec_format(model_loc, binary=True)
        # Get dimensions of word vector
        word_dim = word_vectors['the'].shape[0]
        return word_vectors, word_dim

    def build_label_to_vt(self):
        # Convert image label(number) to text and vector
        # Get label-to-text dictionary
        data = _.unpickle_data('meta')
        labels = data[b'fine_label_names']
        # A placeholder for word vectors
        y_vec = np.empty([100, self.word_dim])
        # A placeholder for text
        y_text = []
        for i in range(100):
            """
            Put text and vectors into collection
            if text contain two words, we use latter word only
            (ex: lawn_mower => mower)
            because word2vec only accept single word
            """
            text = labels[i].decode('ascii')
            text = text.split('_')[-1]
            y_vec[i] = self.word_vectors[text]
            y_text.append(text)
        return y_vec, y_text

    def label_to_vector(self, y):
        # Transform entire label set to vector
        y_vec = np.empty([y.shape[0], self.word_dim])
        for a in range(y.shape[0]):
            y_vec[a] = self.label_vec[y[a]]
        return y_vec

    def label_to_text(self, y):
        # Transform entire label set to text
        y_text = []
        for a in range(y.shape[0]):
            y_text.append(self.label_text[y[a]])
        return y_text

    def load_data(self):
        # Load dataset from file
        x_train, y_train = _.load_train_set()
        x_flat_hit, y_flat_hit = _.load_test_set(sample=self.sample_or_not)
        x_zero_shot, y_zero_shot = _.load_zeroshot_set(sample=self.sample_or_not)
        # Some simple preprocess.
        self.x_train = x_train.astype('float32') / 255
        self.x_flat_hit = x_flat_hit.astype('float32') / 255
        self.x_zero_shot = x_zero_shot.astype('float32') / 255
        # Training use vector to calculate hinge loss
        self.y_train = self.label_to_vector(y_train)
        # Evaluate use text to test if correct
        self.y_flat_hit = self.label_to_text(y_flat_hit)
        self.y_zero_shot = self.label_to_text(y_zero_shot)
        print('Data load complete.')

    def do_freeze_layers(self, model):
        # freeze pre trained layers
        for li in range(len(model.layers)):
            layer = model.layers[li]
            layer.trainable = False
            model.layers[li] = layer
        print('Visual model layers freezed.')
        return model

    def build_model(self):
        # Load visual model
        model = load_model(self.visual_model_path)
        """
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        ###Pop these three layers from previous training
        """
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        if self.freeze_layers:
            model = self.do_freeze_layers(model)
        # Add transformation layer
        model.add(Dense(self.word_dim, use_bias=True, name='dense_trans'))
        # Set transformation layer as network output
        #model.outputs = [model.layers[-1].output]
        #model.layers[-1].outbound_nodes = []
        # Print network structure
        model.summary()
        return model

    def hinge_loss(self, y_true, y_pred):
        # Custom loss function
        margin = 0.1
        # loop counter
        i = tf.constant(0)
        # loop condition function
        c = lambda i, _: tf.less(i, tf.shape(y_true)[0])
        outer_sum_loss = tf.constant(0.0)
        def process_ele(i, outer_sum_loss):
            # Get a subtensor from batch
            y_true_one = y_true[i]
            y_pred_one = y_pred[i]
            # Stack margin to a num_class*1 matrix
            margin_stack = tf.reshape(tf.stack([tf.constant(0.1)] * self.num_classes), [self.num_classes, 1])
            # Stack true label to a word_dim*num_class matrix and transpose it
            y_true_one_stack = tf.stack([tf.transpose(y_true_one)] * self.num_classes)
            # Reshape predict from (word_dim,) to (word_dim,1)
            y_pred_one_t = tf.reshape(y_pred_one, [self.word_dim, 1])
            # Calculate loss
            r = margin_stack - tf.matmul(y_true_one_stack, y_pred_one_t) + tf.matmul(self.label_vec_tensor, y_pred_one_t)
            # Summation
            # We did not exclude true label inside summation, so we subtract extra margin
            sum_inner_loss = tf.reduce_sum(K.relu(r)) - margin
            # Return counter++ and accumulated loss
            return tf.add(i, 1), tf.add(outer_sum_loss, sum_inner_loss)
        
        _, outer_sum_loss = tf.while_loop(c, process_ele, [i, outer_sum_loss])
        # Return average loss over batch
        return outer_sum_loss / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

    def get_eval_set(self):
        # Prepare data for evaluate class
        x, y = {}, {}
        x['flat'], y['flat'] = self.x_flat_hit, self.y_flat_hit
        x['zeroshot'], y['zeroshot'] = self.x_zero_shot, self.y_zero_shot
        return x, y

    def get_train_batch(self):
        # A function yield a batch of training data
        while 1:
            counter = 0
            len_train = self.x_train.shape[0]
            while (counter + 1) * self.batch_size < len_train:
                start = counter * self.batch_size
                end = (counter + 1) * self.batch_size
                yield self.x_train[start:end], self.y_train[start:end]
            yield self.x_train[counter * self.batch_size:len_train], self.y_train[counter * self.batch_size:len_train]

    def train_ops(self):
        # We use Adam as optimizer
        opt = keras.optimizers.Adam(lr=self.lr)
        # Introduce loss function as our custom loss function
        self.model.compile(loss = self.hinge_loss, optimizer = opt)
        # Use TensorBoard to record training process
        tbcb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        eval_x, eval_y = self.get_eval_set()
        self.model.fit_generator(self.get_train_batch(),
                            steps_per_epoch=len(self.x_train) // self.batch_size + 1,
                            epochs=self.epochs,
                            callbacks=[evaluate(self.word_vectors, self.model, eval_x, eval_y), tbcb])