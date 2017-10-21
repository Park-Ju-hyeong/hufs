import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# 데이터가 들어있는 dir를 입력하세요
data_dir = 'data/'

# output이 생성될 dir를 만드세요
out_dir = 'out/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


def data_propre(params):
    """
    데이터를 불러오고 scaling 해서
    input과 output으로 return 합니다.
    """
    credit = pd.read_csv(data_dir + "credit44_sc.csv", encoding="cp949")
    credit_label = credit[credit["target"] == params.label]

    del credit_label["target"]

    print(credit_label.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(credit_label)
    credit_label_sc = scaler.transform(credit_label) * 2
    #
    # scaler = MinMaxScaler()
    # scaler.fit(credit_label)
    # credit_label_sc = scaler.transform(credit_label)

    return credit_label_sc, credit_label


class DataDistribution(object):
    def __init__(self):
        pass

    def sample(self, N):
        """
        미니배치 함수를 만듭니다.
        """
        x = credit_gb[np.random.choice(credit_gb.shape[0], N, replace=True)]
        return tf.constant(x, dtype=tf.float32)


def standard_normal(shape, **kwargs):
    """
    다변량 정규분포를 생성하는 함수입니다.
    """
    return st.StochasticTensor(
        ds.MultivariateNormalDiag(loc=tf.zeros(shape), scale_diag=tf.ones(shape), **kwargs))


def lrelu(x, leak=0.03, name="lrelu"):
    """
    leaky relu 함수입니다.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generative_network(batch_size, input_dim, n_hidden, n_layer):
    """
    generator 함수입니다.
    """
    z = data_batch.sample(batch_size)
    with tf.variable_scope("generative"):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=lrelu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer)
            p = slim.fully_connected(h, input_dim, activation_fn=None)
            x = st.StochasticTensor(ds.Normal(p * tf.ones(input_dim), 1 * tf.ones(input_dim), name="p_x"))

    return [x, z]


def inference_network(x, latent_dim, n_hidden, n_layer, eps_dim):
    """
    inference 함수입니다.
    """
    eps = standard_normal([x.get_shape()[0], eps_dim], name="eps").value()
    c = tf.concat([x, eps], 1)
    with tf.variable_scope("inference"):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=lrelu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            h = slim.stack(c, slim.fully_connected, [n_hidden] * n_layer)
            z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")

    return z


def discriminative_network(x, z, n_hidden):
    """
    discriminator 함수입니다.
    """
    h = tf.concat([x, z], 1)
    # , normalizer_fn=slim.batch_norm
    h = slim.fully_connected(h, n_hidden, activation_fn=lrelu)
    h = slim.dropout(h, 0.5)
    h = slim.fully_connected(h, n_hidden, activation_fn=lrelu)
    h = slim.dropout(h, 0.5)
    h = slim.fully_connected(h, n_hidden, activation_fn=lrelu)
    h = slim.dropout(h, 0.5)
    d = slim.fully_connected(h, 1, activation_fn=None)

    return tf.squeeze(d, squeeze_dims=[1])


class GAN(object):
    def __init__(self, params):
        self.x = tf.random_normal([params.batch_size, params.input_dim])

        self.p_x, self.p_z = generative_network(
            params.batch_size,
            params.input_dim,
            params.hidden_size,
            params.num_layer
        )

        self.q_z = inference_network(
            self.x,
            params.output_dim,
            params.hidden_size,
            params.num_layer,
            params.eps_dim
        )

        with tf.variable_scope('discriminator'):
            self.log_d_prior = discriminative_network(
                self.p_x, self.p_z, n_hidden=params.hidden_size)

        with tf.variable_scope('discriminator', reuse=True):
            self.log_d_posterior = discriminative_network(
                self.x, self.q_z, n_hidden=params.hidden_size)

        self.disc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.log_d_posterior, labels=tf.ones_like(self.log_d_posterior)) +
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.log_d_prior, labels=tf.zeros_like(self.log_d_prior))
        )

        self.recon_likelihood_prior = self.p_x.distribution.log_prob(self.x)

        self.recon_likelihood = tf.reduce_sum(graph_replace(
            self.recon_likelihood_prior, {self.p_z: self.q_z}), [1])

        self.gen_loss = tf.reduce_mean(self.log_d_posterior) - \
                        tf.reduce_mean(self.recon_likelihood)

        self.qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
        self.pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
        self.dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        self.opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)

        self.train_gen_op = self.opt.minimize(self.gen_loss, var_list=self.qvars + self.pvars)
        self.train_disc_op = self.opt.minimize(self.disc_loss, var_list=self.dvars)


def train(model, params):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):

            loss, _, _ = sess.run([[model.disc_loss, model.gen_loss],
                                   model.train_gen_op, model.train_disc_op])
            if step % params.log_every == 0:
                main = '{}\t Discriminator: {:.4f}\t Generator: {:.4f}'
                print(main.format(step, loss[0], loss[1]))


        samps_plot = samples(model, sess, params)
        if params.save:
            scaler = preprocessing.StandardScaler()
            scaler.fit(credit_good)
            if params.label == 0:
                g = gen_samples(model, sess, params, N=params.num_gen)
                samps = scaler.inverse_transform(g / 2)
                samps = np.round(samps, 0).astype(np.int)
                main = out_dir + "credit_gan_good_3dim"
                np.savetxt('{}.csv'.format(main), samps, delimiter=',')
                plot_distributions(samps_plot, main, save=True)

            if params.label == 1:
                g = gen_samples(model, sess, params, N=params.num_gen)
                samps = scaler.inverse_transform(g / 2)
                samps = np.round(samps, 0).astype(np.int)
                main = out_dir + "credit_gan_bad_3dim"
                np.savetxt('{}.csv'.format(main), samps, delimiter=',')
                plot_distributions(samps_plot, main, save=True)

        else:
            plot_distributions(samps_plot)


def gen_samples(model, sess, params, N):
    """
    학습이 끝난 후 gen 데이터를 생성합니다.
    """
    g = np.zeros((N, params.output_dim))
    for i in range(N // params.batch_size):
        g[params.batch_size * i:params.batch_size * (i + 1), :] = sess.run(model.q_z)
    return g


def samples(model, sess, params):
    """
    real 과 gen 데이터를 추출해 histogram 분포를 가져옵니다.
    """
    bins = np.linspace(-5, 5, 50)

    d = np.zeros((50000, params.output_dim))
    for i in range(50000 // params.batch_size):
        d[params.batch_size * i:params.batch_size * (i + 1), :] = sess.run(model.p_z)
    pd = [np.histogram(d[:, i], bins=bins, density=True)[0] for i in range(params.output_dim)]

    g = np.zeros((50000, params.output_dim))
    for i in range(50000 // params.batch_size):
        g[params.batch_size * i:params.batch_size * (i + 1), :] = sess.run(model.q_z)
    pg = [np.histogram(g[:, i], bins=bins, density=True)[0] for i in range(params.output_dim)]

    return pd, pg


def plot_distributions(sample, save_path=None, save=False):
    """
    histogram을 그립니다.
    """
    pd, pg = sample

    p_x = np.linspace(-5, 5, len(pd[0]))
    f, ax = plt.subplots(4, 11, figsize=(25, 10))
    f.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, axes in enumerate(ax.flat):
        axes.set_ylim(0, 1.5)
        axes.plot(p_x, pd[i], label='real data')
        axes.plot(p_x, pg[i], label='generated data')
        # axes.set_title("{}".format(iris.feature_names[i]))

    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()

    if save:
        plt.savefig("{}.png".format(save_path))
    else:
        plt.show()


def main(args):
    model = GAN(args)
    train(model, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=5000,
                        help='the number of training steps to take')
    parser.add_argument('--input-dim', type=int, default=254,
                        help='input size')
    parser.add_argument('--output-dim', type=int, default=44,
                        help='output size')
    parser.add_argument('--eps-dim', type=int, default=1,
                        help='eps size')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='MLP hidden size')
    parser.add_argument('--num-layer', type=int, default=5,
                        help='number of hidden layer')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=200,
                        help='print loss after this many steps')
    parser.add_argument('--num-gen', type=int, default=10000,
                        help='number of generative data')
    parser.add_argument('--save', type=bool, default=True,
                        help='save plot and csv for generated data')
    parser.add_argument('--label', type=int, default=0,
                        help='select good or bad')

    return parser.parse_args()


if __name__ == '__main__':
    credit_gb, credit_good = data_propre(parse_args())
    data_batch = DataDistribution()
    main(parse_args())