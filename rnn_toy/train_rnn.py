'''training simple rnn model'''
#-*- coding:utf-8 -*-

from rnn import SimpleRnn
from data_loader import LmDataLoader
from evaluation import Entropy

def train_rnn(train_corpus, valid_corpus, batch_size, vocab_size, hiden_size, rounds, alpha, bptt):
    '''train a rnn model'''
    simple_rnn = SimpleRnn(batch_size, vocab_size, hiden_size, vocab_size, alpha=alpha, bptt=bptt)
    train_data_loader = LmDataLoader(train_corpus, batch_size)
    valid_data_loader = LmDataLoader(valid_corpus, batch_size)

    valid_evaluate = Entropy(batch_size, valid_data_loader, simple_rnn)

    for i in xrange(rounds):
        print "round: ", i
        instances = 0
        train_data_loader.reset()
        for input_batch, target_batch in train_data_loader:
            instances += input_batch.shape[0]
            if instances % 10000 == 0:
                print "processed instances: ", instances
            simple_rnn.train_one_batch(input_batch, target_batch)
        simple_rnn.reset_hiden_state()
        print "valid data entropy: ", valid_evaluate.cal_entropy()
        


def main():
    '''training test'''
    train_corpus = "./tests/small_train.txt"
    valid_corpus = "./tests/small_valid.txt"
    batch_size = 25
    vocab_size = 1396
    hiden_size = 50
    rounds = 10
    alpha = 0.1
    bptt = 0
    train_rnn(train_corpus, valid_corpus, batch_size, vocab_size, hiden_size, rounds, alpha, bptt)

if __name__ == '__main__':
    main()
    