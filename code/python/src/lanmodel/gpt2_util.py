import gpt_2_simple as gpt2
import sys

def encode_and_compress(inFile):
    gpt2.encode_dataset(inFile)


if __name__ == "__main__":
    if sys.argv[1]=="encode":
        encode_and_compress(sys.argv[2])