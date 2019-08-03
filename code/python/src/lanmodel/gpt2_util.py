import gpt_2_simple as gpt2
import sys

def encode_and_compress(inFile):
    gpt2.encode_dataset(inFile)

def fine_tune(inFile):
    model_name = "117M"
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/117M/

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  inFile,
                  model_name=model_name,
                  steps=1000,
                  save_every=100)  # steps is max number of training steps

    gpt2.generate(sess)

if __name__ == "__main__":
    if sys.argv[1]=="encode":
        encode_and_compress(sys.argv[2])
    elif sys.argv[1]=="ft":
        fine_tune(sys.argv[2])