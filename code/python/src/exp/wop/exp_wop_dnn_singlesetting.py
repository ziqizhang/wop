from distutils.util import strtobool

from exp.wop import exp_wop_dnn as dnn_exp
import sys

if __name__ == "__main__":
    dnn_exp.run_single_setting(sys.argv[1], sys.argv[2],
                               strtobool(sys.argv[3]),
                               strtobool(sys.argv[4]))
