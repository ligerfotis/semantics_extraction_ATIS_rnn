from tensorflow.python.ops import rnn
from metrics.accuracy import conlleval
from utils import tools
import tensorflow as tf
from data import load
import numpy as np
import subprocess
import time
import os

folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder): os.mkdir(folder)

# Hyper Parameters
context_size = 3
hidden_size = 300
embedding_size = 300
init_scale = 1
batch_size = 1
verbose = 0
seed = 345
epochs_num = 40
lr = 0.0001200
decay = True
num_layers = 2
dropout = .5
istraining = True
vocsize = 572
num_classes = 127
preTraining = True

tf.reset_default_graph()
#input and labels placeholders
x_input = tf.placeholder(tf.int32, [batch_size,None], name="input_x")
y_labels = tf.placeholder(tf.int32,[None], name="labels")
embedding = tf.Variable(tf.random_uniform([vocsize + 1, embedding_size], -init_scale, init_scale, seed=seed),
                            dtype=tf.float32, trainable=True)

#The pretraining model for word embeddings
def preTrainModel():

    inputs = tf.nn.embedding_lookup(embedding, x_input, name="embedding") # shape: (1,5,50)
    W2   = tf.Variable(tf.random_uniform([hidden_size, num_classes],-init_scale, init_scale,tf.float32,seed=seed), name="Weights")
    b2 =  tf.Variable(tf.constant(0.0, shape=[num_classes]), name="Bias")

    # Reshape of RNN layer output
    last = tf.reshape(inputs, [-1, hidden_size])
    Yhat = tf.matmul(last, W2) + b2
    return Yhat

#The RNN Model
def model():
    #saver.restore(sess, "/tmp/model.ckpt")
    #embedding = tf.get_variable("embedding",shape=[vocsize + 1, embedding_size],trainable= False,dtype=tf.float32)

    embedding = tf.Variable(tf.random_uniform([vocsize + 1, embedding_size], -init_scale, init_scale, seed=seed),
                            dtype=tf.float32, trainable=True)
    inputs = tf.nn.embedding_lookup(embedding, x_input, name="emb_input")

    if istraining and dropout<1:
        inputs = tf.nn.dropout(inputs, dropout)
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    if istraining and dropout < 1:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])

    value, states = rnn.dynamic_rnn(cell, inputs, dtype = tf.float32)

    W2   = tf.Variable(tf.random_uniform([hidden_size, num_classes],-init_scale, init_scale,tf.float32,seed=seed), name="Weights")
    #b2 =  tf.Variable(tf.constant(0.0, shape=[num_classes]), name="Bias")

    # Reshape of RNN layer output
    #value = tf.transpose(value, [1, 0, 2])
    #last = tf.gather(value, 2)
    last = tf.reshape(value, [-1, hidden_size])
    #Yhat = tf.matmul(last, W2) + b2
    Yhat = tf.matmul(last, W2)
    return Yhat

#inference function: Softmax Activation
def inference(model):
    return tf.nn.softmax(model)

#loss function: Softmax Cross Entropy
def loss(model):
    SoftCE = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=y_labels)
    return tf.reduce_mean(SoftCE)

#minimize Adam
def train(total_loss, lr):
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.9999)
    goal = opt.minimize(total_loss)
    return goal

#evaluate according to F1 score
#prints predictions to files
def evaluate(sess,model):
    Yhat = inference(model)
    #Return the index with the largest value across axis
    Ypredict = tf.argmax(Yhat, axis=1, output_type=tf.int32)

    # predictions test
    predictions_test = [map(lambda x: idx2label[x],
                            sess.run(Ypredict, feed_dict={x_input:[sentence]}))
                        for sentence in test_lex]
    groundtruth_test = [map(lambda x: idx2label[x], label) for label in test_y]
    words_test = [map(lambda x: idx2word[x], word) for word in test_lex]

    predictions_valid = [map(lambda x: idx2label[x],
                             sess.run(Ypredict, feed_dict={x_input: [sentence]}))
                         for sentence in valid_lex]
    groundtruth_valid = [map(lambda x: idx2label[x], label) for label in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

    # evaluation // compute the accuracy using conlleval.pl
    res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + "current.test.txt")
    res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + "current.valid.txt")

    return res_test,res_valid

#the training session loop
def trainingSession(model,best_f1):
    for epoch in range(epochs_num):
        # Shuffle
        tools.shuffle([train_lex, train_ne, train_y], seed)
        tic = time.time()
        train_loss = np.inf
        # Train
        for i in range(nsentences):
            labels = train_y[i]
            train_x = np.array([train_lex[i]])

            temp_loss, _ = sess.run([total_loss, train_op],
                                    feed_dict={x_input: train_x, y_labels: labels})

            if train_loss > temp_loss:
                train_loss = temp_loss
            if verbose:
                print("[Learning] Epoch %i >> %2.2f%%" % (epoch + 1, (i + 1) * 100. / nsentences),
                      "completed in %.2f (sec) <<\r" % (time.time() - tic))
        print("[Learning] Epoch %i Loss %2.2f" % (epoch + 1, train_loss * 100000))

        # =============================#
        #           Evaluation         #
        #                              #
        #   back into the real world   #
        #         idx -> words         #
        # =============================#
        print("Evaluating...")
        istraining = False
        res_test, res_valid = evaluate(sess, model)
        istraining = True

        if res_valid['f1'] > best_f1:
            # Save the variables to disk.
            #save_path = saver.save(sess, "/tmp/model.ckpt")
            #print("Model saved in path: %s" % save_path)
            best_f1 = res_valid['f1']
            if 1:
                print("NEW BEST: epoch", epoch + 1,
                      "valid F1", res_valid['f1'],
                      "best test F1", res_test['f1'], " " * 20)

            vf1, vp, vr = res_valid['f1'], res_valid['p'], res_valid['r']
            tf1, tp, tr = res_test['f1'], res_test['p'], res_test['r']
            best_epoch = epoch
            subprocess.call(['mv', folder + "current.test.txt", folder + "best.test.txt"])
            subprocess.call(['mv', folder + "current.valid.txt", folder + "best.valid.txt"])
        else:
            print()

        #if preTraining and train_loss<0.5:
        #    break

    # print the best result
    print('BEST RESULT: epoch', best_epoch + 1,
          'valid F1', vf1, 'best test F1', tf1, 'with the model', folder)
    return best_f1

#initialize tf Session
#5-fold cross validation
with tf.Session() as sess:
    #preModel = preTrainModel()
    #saver = tf.train.Saver({"trained_embedding": embedding})
    #Embedding training
    #trainingSession(preModel)
    #preTraining = False

    model = model()
    total_loss = loss(model)
    train_op = train(total_loss, lr)
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Starting Training... Please Wait...")
    best_f1 = -np.inf
    current_lr = lr
    best_epoch = 0
    for fold in range(0,5):
        # load the dataset
        train_set, valid_set, test_set, dic = load.atisfold(fold)
        idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
        idx2word = dict((k, v) for v, k in dic['words2idx'].items())

        train_lex, train_ne, train_y = train_set
        valid_lex, valid_ne, valid_y = valid_set
        test_lex, test_ne, test_y = test_set

        # maxSentenceLength = np.amax([len(i) for i in train_lex+train_ne+train_y])
        nsentences = len(train_lex)
        print("Fold ",fold+1)
        best_f1 = trainingSession(model,best_f1)



