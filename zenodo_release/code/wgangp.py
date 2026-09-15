"""
Conditional WGAN-GP for tabular breath VOC profiles.

This is a cleaned, seeded, batched reimplementation of the model in
sip_model_gen (main.py + CTGAN.py). The architecture, preprocessing,
optimiser settings and hyperparameters are unchanged from the released
gen_v22 configuration; the class is renamed from CTGAN to CondWGANGP
because the model is a conditional WGAN-GP, not Xu et al.'s CTGAN.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import numpy as np, tensorflow as tf, keras, json, random
from common import Preprocessor, CLASSES
from keras import Model, layers
from sklearn.preprocessing import MinMaxScaler

LATENT_DIM, NUM_CLASSES = 128, 3
EPOCHS, BATCH_SIZE = 2000, 32
N_CRITIC, GP_WEIGHT = 5, 10.0
LR, BETA_1, BETA_2 = 1e-4, 0.0, 0.9
CLASSES = ["asthma", "bronchiectasis", "copd"]


def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def build_generator(n_voc, name="gen"):
    z = layers.Input(shape=(LATENT_DIM,)); c = layers.Input(shape=(NUM_CLASSES,))
    x = layers.Concatenate()([z, c])
    for u in (256, 512, 512, 256):
        x = layers.Dense(u)(x); x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dense(n_voc, activation='sigmoid')(x)
    return Model([z, c], x, name=name)


def build_critic(n_voc, name="dis"):
    v = layers.Input(shape=(n_voc,)); c = layers.Input(shape=(NUM_CLASSES,))
    x = layers.Concatenate()([v, c])
    for u in (512, 256, 128):
        x = layers.Dense(u)(x); x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.LayerNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(64)(x); x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dense(1)(x)                       # linear critic output
    return Model([v, c], x, name=name)


class CondWGANGP(Model):
    def __init__(self, critic, generator, latent_dim=LATENT_DIM,
                 n_critic=N_CRITIC, gp_weight=GP_WEIGHT):
        super().__init__()
        self.critic, self.generator = critic, generator
        self.latent_dim, self.n_critic, self.gp_weight = latent_dim, n_critic, gp_weight

    def compile(self, d_optimizer, g_optimizer, **kw):
        super().compile(**kw)
        self.d_optimizer, self.g_optimizer = d_optimizer, g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.gp_metric     = keras.metrics.Mean(name="gp")

    @property
    def metrics(self): return [self.d_loss_metric, self.g_loss_metric, self.gp_metric]

    def _gp(self, real, fake, labels):
        b = tf.shape(real)[0]
        alpha = tf.random.uniform([b, 1], 0.0, 1.0)
        inter = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = self.critic([inter, labels], training=True)
        g = t.gradient(pred, inter)
        norm = tf.sqrt(tf.reduce_sum(tf.square(g), axis=1) + 1e-8)
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        real, labels = data
        b = tf.shape(real)[0]
        for _ in range(self.n_critic):
            z = tf.random.normal((b, self.latent_dim))
            with tf.GradientTape() as tape:
                fake = self.generator([z, labels], training=True)
                d_loss = (tf.reduce_mean(self.critic([fake, labels], training=True))
                          - tf.reduce_mean(self.critic([real, labels], training=True)))
                gp = self._gp(real, fake, labels)
                d_loss = d_loss + self.gp_weight * gp
            self.d_optimizer.apply_gradients(
                zip(tape.gradient(d_loss, self.critic.trainable_variables),
                    self.critic.trainable_variables))
        z = tf.random.normal((b, self.latent_dim))
        with tf.GradientTape() as tape:
            g_loss = -tf.reduce_mean(
                self.critic([self.generator([z, labels], training=True), labels], training=True))
        self.g_optimizer.apply_gradients(
            zip(tape.gradient(g_loss, self.generator.trainable_variables),
                self.generator.trainable_variables))
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.gp_metric.update_state(gp)
        return {m.name: m.result() for m in self.metrics}


def train(X_raw, y_idx, epochs=EPOCHS, seed=0, verbose=0):
    """Fit preprocessor + conditional WGAN-GP on raw VOC values."""
    set_seed(seed)
    pre = Preprocessor().fit(X_raw)
    Xn = pre.transform(X_raw).astype(np.float32)
    Y  = keras.utils.to_categorical(y_idx, NUM_CLASSES).astype(np.float32)
    ds = (tf.data.Dataset.from_tensor_slices((Xn, Y))
          .shuffle(1000, seed=seed).batch(BATCH_SIZE))
    gan = CondWGANGP(build_critic(Xn.shape[1]), build_generator(Xn.shape[1]))
    gan.compile(d_optimizer=keras.optimizers.Adam(LR, BETA_1, BETA_2),
                g_optimizer=keras.optimizers.Adam(LR, BETA_1, BETA_2))
    hist = gan.fit(ds, epochs=epochs, verbose=verbose)
    return gan.generator, pre, hist.history


def sample(generator, pre, n_per_class, seed=0, batch=512):
    """Batched, seeded sampling -> raw-scale VOC matrix + integer labels."""
    set_seed(seed)
    out, labs = [], []
    for i in range(NUM_CLASSES):
        rem = n_per_class
        while rem > 0:
            b = min(batch, rem)
            z = tf.random.normal((b, LATENT_DIM))
            c = tf.repeat(tf.one_hot([i], NUM_CLASSES), b, axis=0)
            out.append(generator([z, c], training=False).numpy())
            labs += [i] * b
            rem -= b
    return pre.inverse(np.vstack(out)), np.array(labs)
