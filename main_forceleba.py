import tensorflow as tf

from utils import mkdir_p
import numpy as np
from vaegan import vaegan
from utils import CelebA

flags = tf.app.flags

flags.DEFINE_integer("OPER_FLAG" , 0, "the flag of opertion")
flags.DEFINE_integer("extend" , 0, "contional value y")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./vaeganCeleba/logs{}/celeba_test2".format(FLAGS.OPER_FLAG)
    infogan_checkpoint_dir = "./model_vaegan2/model.ckpt"
    sample_path = "./vaeganCeleba2/sample1"

    mkdir_p(root_log_dir)
    mkdir_p(infogan_checkpoint_dir)
    mkdir_p(sample_path)

    model_path = [infogan_checkpoint_dir]

    batch_size = 64
    max_epoch = 10

    #for mnist train 62 + 2 + 10
    sample_size = 1024

    dis_learn_rate = 0.0003
    gen_learn_rate = 0.0003

    OPER_FLAG = FLAGS.OPER_FLAG
    data_list = CelebA().load_celebA(is_test= False)
    print "the num of dataset", len(data_list)

    infoGan = vaegan(batch_size = batch_size, max_epoch = max_epoch,
                      model_path = model_path, data = data_list,
                      network_type = "celebA",sample_size = sample_size,
                      sample_path = sample_path , log_dir = root_log_dir , gen_learning_rate = gen_learn_rate, dis_learning_rate=dis_learn_rate , info_reg_coeff=1.0)

    if OPER_FLAG == 0:

        infoGan.build_model_infoGan()
        infoGan.train()








