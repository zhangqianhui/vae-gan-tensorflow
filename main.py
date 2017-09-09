import tensorflow as tf

from utils import mkdir_p
from vaegan import vaegan
from utils import CelebA

flags = tf.app.flags

flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_epoch" , 60, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 128, "the dim of latent code")
flags.DEFINE_integer("learn_rate_init" , 0.0003, "the init of learn rate")
flags.DEFINE_string("path" , '/home/jichao/data/celebA', "tthe dataset directory")
flags.DEFINE_integer("operation", 0, "the init of learn rate")


FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./vaeganlogs/logs/celeba_test"
    vaegan_checkpoint_dir = "./model_vaegan/model.ckpt"
    sample_path = "./vaeganSample/sample"

    mkdir_p(root_log_dir)
    mkdir_p(vaegan_checkpoint_dir)
    mkdir_p(sample_path)

    model_path = vaegan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_epoch = FLAGS.max_epoch
    latent_dim = FLAGS.latent_dim

    learn_rate_init = FLAGS.learn_rate_init

    data_list = CelebA().load_celebA(image_path=FLAGS.path)
    print "the num of dataset", len(data_list)

    vaeGan = vaegan(batch_size = batch_size, max_epoch = max_epoch,
                      model_path = model_path, data = data_list, latent_dim = latent_dim,
                      sample_path = sample_path , log_dir = root_log_dir , learnrate_init=learn_rate_init)

    if FLAGS.operation == 0:

        vaeGan.build_model_vaegan()
        vaeGan.train()

    else:

        vaeGan.build_model_vaegan()
        vaeGan.test()









