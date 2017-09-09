import tensorflow as tf

from utils import mkdir_p
from vaegan import vaegan
from utils import CelebA

flags = tf.app.flags

flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_epoch" , 60, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 128, "the dim of latent code")
flags.DEFINE_integer("learn_rate_init" , 0.0003, "the init of learn rate")


FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./vaeganCeleba/logs/celeba_test2"
    infogan_checkpoint_dir = "./model_vaegan2/model.ckpt"
    sample_path = "./vaeganCeleba2/sample"

    mkdir_p(root_log_dir)
    mkdir_p(infogan_checkpoint_dir)
    mkdir_p(sample_path)

    model_path = infogan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_epoch = FLAGS.max_epoch
    latent_dim = FLAGS.latent_dim
    learn_rate_init = FLAGS.learn_rate_init

    data_list = CelebA().load_celebA(is_test= False)
    print "the num of dataset", len(data_list)

    vaeGan = vaegan(batch_size = batch_size, max_epoch = max_epoch,
                      model_path = model_path, data = data_list, latent_dim = latent_dim,
                      sample_path = sample_path , log_dir = root_log_dir , learnrate_init=learn_rate_init)

    vaeGan.build_model_vaegan()
    vaeGan.train()








