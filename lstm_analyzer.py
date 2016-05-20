import numpy as np
from numpy import array
import pickle
import theano
import theano.tensor as T
import lasagne
import datetime
import logging
import re
import os
import collections

import config
from analyzer import Analyzer

logging.basicConfig(filename=os.path.join(config.WORK_DIR, 'log', 'lstm_debug.log'), level=logging.DEBUG)
logging.basicConfig(filename=os.path.join(config.WORK_DIR, 'log', 'lstm_info.log'), level=logging.INFO)
TOKEN_PATTERN = '[a-zA-Z0-9\_]+|[\+\-\*\/\>\<\=]+'
NUM_EPOCHS = 100
SEQ_LENGTH = 20
N_HIDDEN = 64
PREDICTION_LENGTH = 30

CODE_SNIPPET_1 = '''
                JSONObject js = new JSONObject(value.toString());
                if (js.has("user_id")) {
                    // is a blog
                    userID.set(js.getString("user_id"));
                    context.write(userID, new Text("t")); // text
                '''
CODE_SNIPPET_2 = '''
                String s = tokenizer.nextToken();
                JSONObject js = new JSONObject(s);

                if (js.has("user_id"))                      // 
                {
                    try {
                    userId.set(js.optString("user_id"));
                    } catch (JSONException e)
                    {
                        System.out.println(s);
                    }
                    context.write(userId, one);
'''

class LSTM_Analyzer(Analyzer):
    def __init__(self, data_accessor):
        self.data_accessor = data_accessor

    def _build_network(self, input_var=None):
        print("Building network")
        logging.debug('Building network ...')
   
        # First, we build the network, starting with an input layer
       	# Recurrent layers expect input of shape
        # (batch size, SEQ_LENGTH, num_features)
        network = lasagne.layers.InputLayer(shape=(None, None, self.vocab_size), input_var=input_var)

        # We now build the LSTM layer which takes l_in as the input layer
        # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 
        network = lasagne.layers.LSTMLayer(network, N_HIDDEN, nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.LSTMLayer(network, N_HIDDEN, nonlinearity=lasagne.nonlinearities.tanh)

        # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
        # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
        # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
        network = lasagne.layers.SliceLayer(network, -1, 1)

        # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
        # The output of this stage is (batch_size, vocab_size)
        network = lasagne.layers.DenseLayer(network, num_units=self.vocab_size, W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
        return network

    def _train_model(self):
        print("Preparing...")
        logging.info("Preparing...")
        input_var = T.tensor3('inputs')
        target_var = T.ivector('targets')
        network = self._build_network(input_var)

        # Network Output
        network_output = lasagne.layers.get_output(network)

        # Cross entropy function
        cost = lasagne.objectives.categorical_crossentropy(network_output, target_var)

        # Loss function: Cross entropy and regularization
        loss = cost.mean() + 1e-4 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

        # Update function
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.9)

        # Training Function
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Predict the output
        predict = theano.function([input_var], T.argmax(network_output, axis=1))

        # Cost function
        cross_entro = theano.function([input_var, target_var], cost.mean())

        print("Training start.")
        for epoch in range(NUM_EPOCHS):
            loss_cnt = 0.
            for input_batch, target_batch in self._training_data():
                loss_cnt += train_fn(input_batch, target_batch)
            print("Epoch {}: Loss {}".format(epoch+1, loss_cnt/self.SIZE_TRAINING_DATA))
            logging.info("Epoch {}: Loss {}".format(epoch+1, loss_cnt/self.SIZE_TRAINING_DATA))

            loss_cnt = 0.
            for input_batch, target_batch in self._validation_data():
                loss_cnt += cross_entro(input_batch, target_batch)
            print("Epoch {}: Validation Loss {}".format(epoch+1, loss_cnt/self.NUM_VALIDATION_DATA))
            logging.info("Epoch {}: Loss {}".format(epoch+1, loss_cnt/self.NUM_VALIDATION_DATA))

            self.predict(CODE_SNIPPET_1, predict)
            self.predict(CODE_SNIPPET_2, predict)

    def _training_data(self):
        for rank, item in enumerate(self.WORDLIB[0:self.SIZE_TRAINING_DATA]):
        # for rank, item in enumerate(self.WORDLIB[0:10]):
            print("Input {}/{}".format(rank, self.SIZE_TRAINING_DATA))
            inputs = []
            targets = []
            tmp_word_vec = np.zeros((len(item), self.vocab_size))

            for index, wd in enumerate(item):
                tmp_word_vec[index][self.wd2id[wd]] = 1
            for index in range(len(item)-SEQ_LENGTH-1):
                inputs.append(tmp_word_vec[index:index+SEQ_LENGTH])
                targets.append(self.wd2id[item[index+SEQ_LENGTH]])
            yield inputs, targets

    def _validation_data(self):
        for rank, item in enumerate(self.WORDLIB[self.SIZE_TRAINING_DATA:self.TOTAL_DATA_SIZE]):
        # for rank, item in enumerate(self.WORDLIB[10:20]):
            print("Input {}/{}".format(rank, self.TOTAL_DATA_SIZE - self.SIZE_TRAINING_DATA))
            inputs = []
            targets = []
            tmp_word_vec = np.zeros((len(item), self.vocab_size))

            for index, wd in enumerate(item):
                tmp_word_vec[index][self.wd2id[wd]] = 1
            for index in range(len(item)-SEQ_LENGTH-1):
                inputs.append(tmp_word_vec[index:index+SEQ_LENGTH])
                targets.append(self.wd2id[item[index+SEQ_LENGTH]])
            yield inputs, targets

    def _generate_feature(self):
        word_set = self.data_accessor.data_set.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action'] in ['paste']).map(lambda x: x['content'])
        word_set = word_set.map(lambda content: filter(lambda x: ord(x)<128 and ord(x)>0, content))
        word_set = word_set.filter_by(lambda x: len(x) > 1000 and 'import java.io.IOException;' in x)
        # for index, item in enumerate(self.WORDLIB):
        #     with open('data/code_template_{}.txt'.format(index), 'w') as f_out:
        #         f_out.write(item)
        self.WORDLIB = word_set.map(lambda x: re.findall(TOKEN_PATTERN, x))
        self.TOTAL_DATA_SIZE = len(self.WORDLIB)
        self.SIZE_TRAINING_DATA = int(self.TOTAL_DATA_SIZE*0.8)
        self.NUM_VALIDATION_DATA = 0
        for item in self.WORDLIB[self.SIZE_TRAINING_DATA:self.TOTAL_DATA_SIZE]:
            self.NUM_VALIDATION_DATA += len(item)

        self.DICT = collections.defaultdict(lambda: 1)
        for item in self.WORDLIB.flatmap(lambda x: x):
            self.DICT[item] += 1

        self.wd2id = { word:index for index, word in enumerate(self.DICT) }
        self.id2wd = { index:word for index, word in enumerate(self.DICT) }

        self.vocab_size = len(self.DICT)
        print("Dictionary Size: {}".format(self.vocab_size))
        logging.info("Dictionary Size: {}".format(self.vocab_size))

    def predict(self, input_sentence, predict_function):
        print("Input: {}".format(input_sentence))
        logging.info("Input: {}".format(input_sentence))
        tokens = re.findall(TOKEN_PATTERN, input_sentence)
        if (len(tokens) < SEQ_LENGTH):
            return
        tmp_word_vec = np.zeros((len(tokens), self.vocab_size)).astype('float32')
        for index, item in enumerate(tokens):
            tmp_word_vec[index][self.wd2id[item]] = 1.

        inputs, targets = [], []
        for item in tokens:
            for index in range(len(tokens)-SEQ_LENGTH-1):
                inputs.append(tmp_word_vec[index:index+SEQ_LENGTH])
                targets.append(tmp_word_vec[index+SEQ_LENGTH])
        
#         for index, item in enumerate(inputs):
#             prediction = predict_function([item])[0]
#             print('Predicted {}/ Golden {}'.format(self.id2wd[prediction], self.argmax))
#             logging.info('Predicted {}/ Golden {}'.format(self.id2wd[prediction], tokens[index]))
        predict_function(inputs)
        sequence = [item for item in tmp_word_vec[len(tokens)-SEQ_LENGTH:len(tokens)]]
        for index in xrange(PREDICTION_LENGTH):
            next_index = predict_function([sequence[index:index+SEQ_LENGTH]])[0]
            next_vec = np.zeros((self.vocab_size))
            next_vec[next_index] = 1
            sequence.append(next_vec)

        print('Output:')
        logging.info('Output:')

        for item in sequence:
            print('{} '.format(self.id2wd[np.argmax(item)]))
            logging.info('{} '.format(self.id2wd[np.argmax(item)]))

    def prepare(self):
        return

    def analyze(self):
        return