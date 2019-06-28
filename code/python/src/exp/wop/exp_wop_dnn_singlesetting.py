from distutils.util import strtobool

from exp.wop import exp_wop_dnn as dnn_exp
import sys
import os

from numpy.random import seed
seed(1)
os.environ['PYTHONHASHSEED'] = '0'

#embedding_file=/data/embeddings/wop/name_cbow.bin

if __name__ == "__main__":
    overwrite_params = dnn_exp.parse_overwrite_params(sys.argv)
    dnn_exp.run_single_setting(sys.argv[1], sys.argv[2],
                               strtobool(sys.argv[3]),
                               strtobool(sys.argv[4]),
                               overwrite_params=overwrite_params,
                               gensimFormat=True)
