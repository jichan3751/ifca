import argparse
import json
import os
import time
import itertools
import pickle
import copy

import tensorflow as tf

import numpy as np

from util import *
import cifar10

from train_cluster_cifar_tf import *


def main():

    config = get_config()
    config['train_seed'] = config['data_seed']
    print("config:",config)

    exp = TrainCIFARClusterLocal(config)
    exp.setup()
    exp.run()


def get_config():
    arg_seed = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir",type=str,default="output")
    parser.add_argument("--dataset-dir",type=str,default="output")
    parser.add_argument("--num-epochs",type=float,default=300)
    # parser.add_argument("--lr",type=float,default=0.2)
    parser.add_argument("--data-seed",type=int,default=1)
    parser.add_argument("--train-seed",type=int,default=arg_seed)
    parser.add_argument("--config-override",type=str,default="")
    args = parser.parse_args()

    # read config json and update the sysarg
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    args_dict = vars(args)
    config.update(args_dict)

    if config["config_override"] == "":
        del config['config_override']
    else:
        print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config


class TrainCIFARClusterLocal(TrainCIFARCluster):

    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok = True)

        self.result_fname = os.path.join(self.config['project_dir'], 'results_local')
        # self.checkpoint_fname = os.path.join(self.config['project_dir'], 'checkpoint_local')

        self.setup_datasets()
        self.setup_model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)


        set_random_seed(self.config['train_seed'])
        self.initialize_models()
        self.initialize_assign_ops()

        # import ipdb; ipdb.set_trace()

        self.epoch = None
        self.lr = None

    def setup_model(self):
        super().setup_model()
        self.train_transform_op = train_transform2(self.x_tr_pl)
        self.test_transform_op = test_transform2(self.x_tr_pl)



    def find_good_initializer(self):
        pass

    def initialize_models(self):

        m = self.config['m']

        # initialize p times, to get p different sets of weights.

        self.init_op = tf.global_variables_initializer()

        self.model_weights = []
        for m_i in range(m):
            self.sess.run(self.init_op)
            weights = self.get_model_weights()
            self.model_weights.append(weights)

    def run(self):
        TRAIN_INFER_FULL_NODES = 0

        num_epochs = self.config['num_epochs']
        lr = self.config['lr']

        results = []

        # epoch -1
        self.epoch = -1

        self.find_good_initializer()
        self.set_participating_nodes()


        result = {}
        result['epoch'] = -1

        t0 = time.time()
        res = self.test(train=True, force_full_nodes =TRAIN_INFER_FULL_NODES)
        # res = self.test(train=True)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['train'] = res
        self.print_epoch_stats(res)

        t0 = time.time()
        res = self.test(train=False)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['test'] = res
        self.print_epoch_stats(res)

        results.append(result)

        # this will be used in next epoch

        for epoch in range(num_epochs):
            self.epoch = epoch

            result = {}
            result['epoch'] = epoch

            lr = self.lr_schedule(epoch)
            result['lr'] = lr

            t0 = time.time()
            result['train'] = self.train(lr = lr)
            t1 = time.time()
            train_time = t1-t0

            t0 = time.time()
            # res = self.test(train=True)
            res = self.test(train=True, force_full_nodes =TRAIN_INFER_FULL_NODES)
            t1 = time.time()
            res['infer_time'] = t1-t0
            res['train_time'] = train_time
            res['lr'] = lr
            result['train'] = res
            self.print_epoch_stats(res)

            if epoch % 10 == 0 and epoch != 0:
                t0 = time.time()
                res = self.test(train=False)
                t1 = time.time()
                res['infer_time'] = t1-t0
                result['test'] = res
                self.print_epoch_stats(res)

            results.append(result)

            if epoch % 10 == 0 or epoch == num_epochs - 1 :
                with open(self.result_fname+".pickle", 'wb') as outfile:
                    pickle.dump(results, outfile)
                    print(f'result written at {self.result_fname+".pickle"}')
                # self.save_checkpoint()
                # print(f'checkpoint written at {self.checkpoint_fname}')


    def set_participating_nodes(self):
        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        p_rate = cfg['participation_rate']

        self.participating_nodes = np.random.choice(m, int(m * p_rate), replace = False)
        # self.participating_nodes = list(range(m))

        return self.participating_nodes

    def lr_schedule(self, epoch):
        if self.lr is None:
            self.lr = self.config['lr']

        if epoch != 0 and LR_DECAY:
            self.lr = self.lr * 0.99

        return self.lr

    def test(self, train = True, force_full_nodes = False):

        VERBOSE = 0

        cfg = self.config
        p = cfg['p']
        p_rate = cfg['participation_rate']

        if train:

            # m_test = cfg['m']
            m = cfg['m']
            n = cfg['n']

            num_data = 0
            losses = []
            corrects = []
            for m_i in self.participating_nodes:
                self.put_model_weights(self.model_weights[m_i])
                (X, y) = self.load_node_data(m_i, train=train)
                (loss, correct) = self.sess.run([self.loss, self.num_correct], feed_dict = {self.x_pl:X, self.y_pl:y})

                losses.append(loss)
                corrects.append(correct)
                num_data += X.shape[0]

            loss = np.mean(losses)
            acc = np.sum(corrects) / num_data

        else:

            m_test = cfg['m_test']
            m = cfg['m']
            n = cfg['n']
            p = cfg['p']

            # find out same cluster machines
            cluster_assign_test = self.dataset['test']['cluster_assign']
            cluster_machines_test = [[] for _ in range(p)]
            for m_i in range(m_test):
                p_i = cluster_assign_test[m_i]
                cluster_machines_test[p_i].append(m_i)

            model_losses = []
            model_accs = []
            for m_i in self.participating_nodes: # for each model
                if VERBOSE and m_i % 1 == 0: print(f'test test m_i {m_i}/{m} processing\r', end="")

                self.put_model_weights(self.model_weights[m_i])

                p_i_model = self.dataset['train']['cluster_assign'][m_i]

                # run inference on same distribution machines
                losses = []
                corrects = []
                num_data = 0
                for m_i2 in cluster_machines_test[p_i]:
                    (X, y) = self.load_node_data(m_i2, train=False)
                    (loss, correct) = self.sess.run([self.loss, self.num_correct], feed_dict = {self.x_pl:X, self.y_pl:y})

                    losses.append(loss)
                    corrects.append(correct)
                    num_data += X.shape[0]

                loss = np.mean(losses)
                acc = np.sum(corrects) / num_data

                model_losses.append(loss)
                model_accs.append(acc)

            loss = np.mean(model_losses)
            acc = np.mean(model_accs)

        cluster_assign = []
        cl_ct=[]
        cl_ct_ans=[]

        res = {} # results
        res['loss'] = loss
        res['acc'] = acc
        res['cl_ct'] = cl_ct
        res['cl_ct_ans'] = cl_ct_ans
        res['is_train'] = train

        return res

    def train(self, lr):

        VERBOSE = 0

        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        tau = cfg['tau']
        n = cfg['n']
        batch_size = cfg['batch_size']

        for m_i2, m_i in enumerate(self.participating_nodes):

            self.put_model_weights(self.model_weights[m_i])

            for l_epoch in range(tau): # local epochs
                if VERBOSE and m_i % 1 == 0: print(f'Local update m_i2 {m_i2}/{len(self.participating_nodes)} processing\r', end = "")

                pmt = np.random.permutation(n)
                local_indices_list = create_batches(pmt, batch_size = batch_size)
                node_data_indices = self.dataset['train']['data_indices'][m_i]

                for b_i, local_indices in enumerate(local_indices_list):

                    current_batch_indices = node_data_indices[local_indices]

                    (X_b, y_b) = self.load_data_by_index(current_batch_indices, m_i)


                    fd0 = {
                        self.x_pl:X_b,
                        self.y_pl:y_b,
                        self.lr_pl:self.lr
                    }
                    self.sess.run([self.train_op], feed_dict= fd0)

            self.model_weights[m_i] = self.get_model_weights()


def train_transform2(reshaped_image):
    # copied from cifar10_input.py / distorted_input()

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [tf.shape(reshaped_image)[0], height, width, 3])
    # tf shape gives dynamic shape

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    float_image = tf.clip_by_value(float_image, 0, 1)

    return float_image

def test_transform2(reshaped_image):
    # copied from cifar10_input.py / input()

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    float_image = tf.clip_by_value(float_image, 0, 1)

    return float_image



if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))