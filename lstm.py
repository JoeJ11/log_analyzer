'''
Recurrent network example.  Trains a 2 layered LSTM network to learn
text from a user-provided input file. The network can then be used to generate
text using a short string as seed (refer to the variable generation_phrase).
This example is partly based on Andrej Karpathy's blog
(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and a similar example in the Keras package (keras.io).
The inputs to the network are batches of sequences of characters and the corresponding
targets are the characters in the text shifted to the right by one. 
Assuming a sequence length of 5, a training point for a text file
"The quick brown fox jumps over the lazy dog" would be
INPUT : 'T','h','e',' ','q'
OUTPUT: 'u'

The loss function compares (via categorical crossentropy) the prediction
with the output/target.

Also included is a function to generate text using the RNN given the first 
character.  

About 20 or so epochs are necessary to generate text that "makes sense".

Written by @keskarnitish
Pre-processing of text uses snippets of Karpathy's code (BSD License)
'''

from __future__ import print_function


import numpy as np
from numpy import array
import pickle
import theano
import theano.tensor as T
import lasagne
import urllib2 #For downloading the sample text file. You won't need this if you are providing your own file.
import datetime
import logging
import re

logging.basicConfig(filename='lstm_debug.log',level=logging.DEBUG)
logging.basicConfig(filename='lstm_info.log',level=logging.INFO)

# TOKEN_PATTERN = '[\"\'].+?[\"\']|[a-zA-Z0-9\(\)\[\]]+|[\.\{\}\:\;\=\+\-\*\/]'
TOKEN_PATTERN = '[a-zA-Z0-9\_\(\)\[\]]+|[\.\{\}\:\;\=\+\-\*\/]'
# TOKEN_PATTERN = '[a-zA-Z0-9\_]+'

try:
    with open('feature_training.pkl','rb') as f_in:
        features = pickle.load(f_in)
    print("Length of training data: {}".format(len(features)))
    with open('index_to_word.pkl', 'rb') as f_in:
        ix_to_char = pickle.load(f_in)
    with open('word_to_index.pkl', 'rb') as f_in:
        char_to_ix = pickle.load(f_in)
    with open('feature_validation.pkl', 'rb') as f_in:
        validations = pickle.load(f_in)
    print("Length of validation data: {}".format(len(validations)))

except Exception as e:
    print("Please verify the location of the input file/URL.")
    print("A sample txt file can be downloaded from https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    raise IOError('Unable to Read Text')

# text_phrase = [
#                 'JSONObject', 'js', '=', 'new', 'JSONObject(value', '.', 'toString())', ';', 'if', '(js', '.', 'has(', 'user_id', '))', '{',
#                 'userID', '.', 'set(js', '.', 'getString(', 'user_id', '))', ';', 'context', '.', 'write(userID', 'new', 'Text(', 't', '))', ';']
text_phrase = [
            'String', 'id', 'null', 'String', 'line', 'value', 'toString()', 'JSONObject', 'js', 'null', ';', 'try', 'js', 'new', 'JSONObject', 'line',
            'catch', 'JSONException', 'e', 'e', 'printStackTrace', 'if', 'js', 'has', 'user_id']
generation_phrase = np.zeros(len(text_phrase))
infrequent_table = {}
infrequent_counter = 0
for index, item in enumerate(text_phrase):
    if item in char_to_ix:
        generation_phrase[index] = char_to_ix[item]
    elif item in infrequent_table:
        generation_phrase[index] = char_to_ix[infrequent_table[item]]
    else:
        infrequent_counter += 1
        generation_phrase[index] = char_to_ix['INFREQUENT_{}'.format(infrequent_counter%30)]
        infrequent_table[item] = 'INFREQUENT_{}'.format(infrequent_counter%30)
print(infrequent_table)

#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 
data_size = len(features)
vocab_size = len(ix_to_char)

vecs = np.zeros((data_size, vocab_size), dtype='int32')
for index in range(data_size):
    vecs[index][features[index]] = 1

vecs_val = np.zeros((len(validations), vocab_size), dtype='int32')
for index in range(len(validations)):
    vecs_val[index][validations[index]] = 1

vecs_gen = np.zeros((len(generation_phrase), vocab_size), dtype='int32')
for index in range(len(generation_phrase)):
    vecs_gen[index][generation_phrase[index]] = 1
#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 20

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 64

# Optimization learning rate
LEARNING_RATE = 1

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 128

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 500


def gen_data(p, batch_size = BATCH_SIZE, data=vecs, target=features, return_target=True):
    '''
    This function produces a semi-redundant batch of training samples from the location 'p' in the provided string (data).
    For instance, assuming SEQ_LENGTH = 5 and p=0, the function would create batches of  p = 0 and BATCH_SIZE = 2
    5 characters of the string (starting from the 0th character and stepping by 1 for each semi-redundant batch)
    as the input and the next character as the target.
    To make this clear, let us look at a concrete example. Assume that SEQ_LENGTH = 5,
    If the input string was "The quick brown fox jumps over the lazy dog.",
    For the first data point,
    x (the inputs to the neural network) would correspond to the encoding of 'T','h','e',' ','q'
    y (the targets of the neural network) would be the encoding of 'u'
    For the second point,
    x (the inputs to the neural network) would correspond to the encoding of 'h','e',' ','q', 'u'
    y (the targets of the neural network) would be the encoding of 'i'
    The data points are then stacked (into a three-dimensional tensor of size (batch_size,SEQ_LENGTH,vocab_size))
    and returned. 
    Notice that there is overlap of characters between the batches (hence the name, semi-redundant batch).
    '''
    x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        x[n] = data[p+n:p+n+SEQ_LENGTH]
        if(return_target):
            y[n] = target[p+n+SEQ_LENGTH]
    return x, y


def main(num_epochs=NUM_EPOCHS):
    input_var = T.tensor3('inputs')

    l_out = build_network(input_var)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
    loss = cost + 1e-4 * lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l2)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    logging.debug('Computing updates ...')
    updates = lasagne.updates.adadelta(loss, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    logging.debug('Compiling functions')
    train = theano.function([input_var, target_values], loss, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([input_var, target_values], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict. 

    def try_it_out(N=30):
        '''
        This function uses the user-provided string "generation_phrase" and current state of the RNN generate text.
        The function works in three steps:
        1. It converts the string set in "generation_phrase" (which must be over SEQ_LENGTH characters long) 
           to encoded format. We use the gen_data function for this. By providing the string and asking for a single batch,
           we are converting the first SEQ_LENGTH characters into encoded form. 
        2. We then use the LSTM to predict the next character and store it in a (dynamic) list sample_ix. This is done by using the 'probs'
           function which was compiled above. Simply put, given the output, we compute the probabilities of the target and pick the one 
           with the highest predicted probability. 
        3. Once this character has been predicted, we construct a new sequence using all but first characters of the 
           provided string and the predicted character. This sequence is then used to generate yet another character.
           This process continues for "N" characters. 
        To make this clear, let us again look at a concrete example. 
        Assume that SEQ_LENGTH = 5 and generation_phrase = "The quick brown fox jumps". 
        We initially encode the first 5 characters ('T','h','e',' ','q'). The next character is then predicted (as explained in step 2). 
        Assume that this character was 'J'. We then construct a new sequence using the last 4 (=SEQ_LENGTH-1) characters of the previous
        sequence ('h','e',' ','q') , and the predicted letter 'J'. This new sequence is then used to compute the next character and 
        the process continues.
        '''

        assert(len(generation_phrase)>=SEQ_LENGTH)
        sample_ix = []
        x,_ = gen_data(len(generation_phrase)-SEQ_LENGTH, 1, data=vecs_gen, target=generation_phrase, return_target=False)

        for i in range(N):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x).ravel())
            # Alternatively, to sample from the distribution instead:
            # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[:,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

        random_snippet = ' '.join([ix_to_char[ix] for ix in generation_phrase]) + ' ' + ' '.join([ix_to_char[ix] for ix in sample_ix])   
        print("----\n %s \n----" % random_snippet)
        logging.info("----\n %s \n----" % random_snippet)


    
    print("Training ...")
    logging.debug('Training ...')
    print("Seed used for text generation is: {}".format([ix_to_char[item] for item in generation_phrase]))
    logging.debug("Seed used for text generation is: {}".format([ix_to_char[item] for item in generation_phrase]))
    p = 0
    epoch_counter = 0
    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):
            try_it_out() # Generate text using the p^th character as the start. 
            
            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                # start_time = datetime.datetime.now()
                x,y = gen_data(p)
                
                #print(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1 
                if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    logging.debug('Carriage Return')
                    p = 0;
                
                avg_cost += train(x, y)
                # print("Time used: {}".format(datetime.datetime.now()-start_time))
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
            logging.info("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
            if int(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE) > epoch_counter:
                epoch_counter += 1
                np.savez('model_{}.npz'.format(epoch_counter), *lasagne.layers.get_all_param_values(l_out))
                val_p = 0
                tmp_cost = 0
                counter = 0
                while(val_p+BATCH_SIZE+SEQ_LENGTH < len(validations)):
                    x,y = gen_data(val_p, data=vecs_val, target=validations)
                    val_p += SEQ_LENGTH + BATCH_SIZE - 1
                    tmp_cost += compute_cost(x,y) 
                    counter += 1
                print("Validation loss = {}".format(tmp_cost / float(counter)))

    except KeyboardInterrupt:
        pass

def build_network(input_var=None):
    print("Building network ...")
    logging.debug('Building network ...')
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)
    network = lasagne.layers.InputLayer(shape=(None, None, vocab_size),input_var=input_var)

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    network = lasagne.layers.LSTMLayer(
        network, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    network = lasagne.layers.LSTMLayer(
        network, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    network = lasagne.layers.SliceLayer(network, -1, 1)

    # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    network = lasagne.layers.DenseLayer(network, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    return network

def predict(model_name):
    input_var = T.tensor3('inputs')
    network = build_network(input_var)
    # Theano tensor for the targets
    target_values = T.ivector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(network)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict. 

    with np.load(model_name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    with open('phrase_lists.txt', 'r') as f_in:
        phrase_list = f_in.read().split('\n')

    feature_list = []
    target_list = []
    for phrase in phrase_list:
        feature, text_feature = generate_feature(phrase)
        feature_list.append(text_feature)

        vecs_feature = np.zeros((len(feature), vocab_size), dtype='int32')
        for index in range(len(feature)):
            vecs_feature[index][feature[index]] = 1
        if not len(feature) >= SEQ_LENGTH:
            continue
        # assert(len(feature)>=SEQ_LENGTH)
        sample_ix = []
        x,_ = gen_data(len(feature)-SEQ_LENGTH, 1, data=vecs_feature, target=feature, return_target=False)

        for i in range(30):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x).ravel())
            # Alternatively, to sample from the distribution instead:
            # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[:,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

        random_snippet = ' '.join([ix_to_char[ix] for ix in feature]) + ' ' + ' '.join([ix_to_char[ix] for ix in sample_ix])   
        print("----\n %s \n----" % random_snippet)
        logging.info("----\n %s \n----" % random_snippet)


def generate_feature(input_string):
    text_phrase = re.findall(TOKEN_PATTERN, input_string)
    generation_phrase = np.zeros(len(text_phrase))
    infrequent_table = {}
    infrequent_counter = 0
    for index, item in enumerate(text_phrase):
        if item in char_to_ix:
            generation_phrase[index] = char_to_ix[item]
        elif item in infrequent_table:
            generation_phrase[index] = char_to_ix[infrequent_table[item]]
        else:
            infrequent_counter += 1
            generation_phrase[index] = char_to_ix['INFREQUENT_{}'.format(infrequent_counter%30)]
            infrequent_table[item] = 'INFREQUENT_{}'.format(infrequent_counter%30)
    print(infrequent_table)
    return generation_phrase, text_phrase

if __name__ == '__main__':
    main()
    # predict('model_50.npz')