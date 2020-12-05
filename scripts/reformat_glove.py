import sys
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

if len(sys.argv) != 1 + 1:
    print("[Usage]:\n\tpython {} [dim]".format(sys.argv[0]))
    sys.exit()

dim = int(sys.argv[1])

glove_file = datapath(os.getcwd() + "/glove/glove.6B.{}d.txt".format(dim))
tmp_file = get_tmpfile(os.getcwd() + 
                       "/glove/glove.6B.{}d.word2vec_format.txt".format(dim))

glove2word2vec(glove_file, tmp_file)