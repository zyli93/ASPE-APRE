import os
import config
import datetime
from models.rinante import RINANTE
from models import modelutils
from utils import utils, datautils
from utils.loggingutils import init_logging
import logging

import tensorflow as tf


def __train(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
            test_sents_file, load_model_file, task):
    init_logging('log/{}-train-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    dst_aspects_file, dst_opinions_file = None, None

    # n_train = 1000
    n_train = -1
    n_tags = 5
    batch_size = 32
    lr = 0.001
    share_lstm = False

    logging.info(word_vecs_file)
    logging.info('load model {}'.format(load_model_file))
    logging.info(test_sents_file)

    print('loading data ...')
    vocab, word_vecs_matrix = datautils.load_word_vecs(word_vecs_file)

    logging.info('word vec dim: {}, n_words={}'.format(word_vecs_matrix.shape[1], word_vecs_matrix.shape[0]))
    train_data, valid_data, test_data = modelutils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file, test_sents_file, test_tok_texts_file,
        vocab, n_train, task)
    print('done')

    model = RINANTE(n_tags, word_vecs_matrix, share_lstm, hidden_size_lstm=hidden_size_lstm,
                    model_file=load_model_file, batch_size=batch_size, lamb=lamb)
    model.train(train_data, valid_data, test_data, vocab, n_epochs=n_epochs, lr=lr, dst_aspects_file=dst_aspects_file,
                dst_opinions_file=dst_opinions_file)
    
    """Zeyu's change, start"""
    # TODO: add save model

    # set working directory and create dir to save model
    WORD_DIR = "./extractors/RINANTE/"
    os.makedirs(WORD_DIR + "saved_model", exist_ok=True)

    # save model
    print("[ZL] saving the model", end=" ")
    saver = tf.train.Saver()
    saver_id = "ep{}-".format(n_epochs)
    with tf.Session() as sess:
        saver.save(sess, saver_id)  # TODO: name an ID

    print("Done!")

    """Zeyu's change, end"""
    


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')

    hidden_size_lstm = 100
    n_epochs = 150
    train_word_embeddings = False

    dataset = 'se15r'
    # dataset = 'se14r'
    # dataset = 'se14l'
    dataset_files = config.DATA_DICT[dataset]

    lamb = 0.001
    lstm_l2_src = False

    if dataset == 'se15r':
        rule_model_file = os.path.join(config.DATA_DIR, 'model-data/pretrain/yelpr9-rest-se15r.ckpt')
        word_vecs_file = os.path.join(config.DATA_DIR, 'model-data/yelp-w2v-sg-se15r.pkl')
    elif dataset == 'se14r':
        rule_model_file = os.path.join(config.DATA_DIR, 'model-data/pretrain/yelpr9-rest-se14r.ckpt')
        word_vecs_file = os.path.join(config.DATA_DIR, 'model-data/yelp-w2v-sg-se14r.pkl')
    else:
        rule_model_file = os.path.join(config.DATA_DIR, 'model-data/pretrain/laptops-amazon-se14l.ckpt')
        word_vecs_file = os.path.join(config.DATA_DIR, 'model-data/laptops-amazon-w2v-se14l.pkl')

    __train(word_vecs_file, dataset_files['train_tok_texts_file'], dataset_files['train_sents_file'],
            dataset_files['train_valid_split_file'], dataset_files['test_tok_texts_file'],
            dataset_files['test_sents_file'], rule_model_file, 'both')
