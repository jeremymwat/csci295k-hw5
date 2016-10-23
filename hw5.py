import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

word_to_int = dict()
int_to_word = dict()
book = []
word_set = set()
word_counter = 0

embedsz = 50
vocabsz = 8001 # 8000 + 1 for UNK
batchsz = 30
num_steps = 20
num_epoch = 1
hsize = 256

def get_batches_with_steps(location, input_list, steps, batch_size):
    in_list = []
    out_list = []
    for i in range(steps):
        in_list.append(input_list[location+i:location+batch_size+i])
        out_list.append(input_list[location+1+i:location+batch_size+1+i])
    return in_list, out_list

with open("tokenized_tails.txt", "r") as train_file:
    for line in train_file:
        line = line.rstrip()
        words = line.split(" ")
        for word in words:
            word = word.lower()
            if word in word_set:
                book.append(word_to_int[word])
            else:
                word_to_int[word] = word_counter
                int_to_word[word_counter] = word
                book.append(word_counter)
                word_counter += 1
                word_set.add(word)

training_corpus = book[:int((.9*len(book)))]
test_corpus = book[int((.9*len(book))):]

inpt = tf.placeholder(tf.int32, [batchsz, num_steps])
outpt = tf.placeholder(tf.int32, [batchsz, num_steps])
dropout_placeholder = tf.placeholder(tf.float32)

E = tf.Variable(tf.random_uniform([vocabsz,embedsz], -1.0, 1.0))

embed = tf.nn.embedding_lookup(E,inpt)
embed_with_dropout = tf.nn.dropout(embed, dropout_placeholder)

BLTM = tf.nn.rnn_cell.BasicLSTMCell(hsize, state_is_tuple=True)

init_sta = BLTM.zero_state(batchsz, tf.float32)

w1 = tf.Variable(tf.truncated_normal([hsize, vocabsz], stddev=0.1))
b1 = tf.Variable(tf.zeros([vocabsz]))

rnn_output, out_state = tf.nn.dynamic_rnn(BLTM, embed_with_dropout, initial_state=init_sta)

rnn_output_reshaped = tf.reshape(rnn_output, [batchsz*num_steps, hsize])

logits = tf.matmul(rnn_output_reshaped, w1)+b1

weights =  tf.ones(batchsz*num_steps)

reshaped_out = tf.reshape(outpt, [batchsz*num_steps])

loss_1 = tf.nn.seq2seq.sequence_loss_by_example([logits], [reshaped_out], [weights], average_across_timesteps=False)

loss = tf.reduce_sum(loss_1)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for e in range(num_epoch):
    total_error = 0
    x = 0
    cur_state = (init_sta.c.eval(session=sess),init_sta.h.eval(session=sess))
    while x < len(training_corpus)-batchsz*num_steps:

        inpt_words, outpt_labels = get_batches_with_steps(x, training_corpus, batchsz, num_steps)

        f_dict = {inpt: inpt_words, outpt: outpt_labels, dropout_placeholder: 0.5,
                init_sta[0]:cur_state[0], init_sta[1]:cur_state[1]}

        x += 1
        _, final_state, err = sess.run([train_step, out_state, loss], feed_dict=f_dict)

        total_error += err/(batchsz*num_steps) # multiply by batchsize to keep calculations for avg perplexity correct

        cur_state = final_state
        if x % (1000/batchsz) == 0:
            print "Train error"
            print np.exp(total_error/(x))

while x < len(test_corpus)-batchsz*num_steps:

    inpt_words, outpt_labels = get_batches_with_steps(x, test_corpus, batchsz, num_steps)

    f_dict = {inpt: inpt_words, outpt: outpt_labels, dropout_placeholder: 0.5,
            init_sta[0]:cur_state[0], init_sta[1]:cur_state[1]}

    x += 1
    final_state, err = sess.run([out_state, loss], feed_dict=f_dict)

    total_error += err/(batchsz*num_steps) # multiply by batchsize to keep calculations for avg perplexity correct

    cur_state = final_state
    if x % (1000/batchsz) == 0:
        print "Test error"
        print np.exp(total_error/(x))

print "FINAL test error"
print np.exp(total_error/(x))