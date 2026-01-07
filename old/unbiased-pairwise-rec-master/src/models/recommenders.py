"""
Recommender models used for the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

def binary_cross_entropy(y_true, y_pred, ind = 0):
        """计算二分类交叉熵损失"""
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)  # 避免 log(0)
        if ind == 0:            
            return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        else:
            return (y_true - y_pred) ** 2


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


@dataclass
class PointwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""
    num_users: np.array
    num_items: np.array
    dim: int
    lam: float
    eta: float
    weight: float = 1.
    clip: float = 0.
    dual_unbias: bool = False
    pow: float = 0.5

    def __post_init__(self, ) -> None:
        """Initialize Class."""
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph')
        self.items = tf.placeholder(tf.int32, [None], name='item_ph')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_ph')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=42))
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=42))
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)

        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.sigmoid(tf.expand_dims(self.logits, 1), name='sigmoid_prediction')

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the unbiased loss for the ideal loss function with binary implicit feedback.
            scores = tf.clip_by_value(self.scores, clip_value_min=self.clip, clip_value_max=1.0)
            orig_scores = tf.pow(scores, 1.0 / self.pow)
            dual_scores = tf.pow(1.0 - orig_scores, self.pow) + 1e-6
            local_losses = (self.labels / scores) * tf.square(1. - self.preds)
            if not self.dual_unbias:
                local_losses += self.weight * (1 - self.labels / scores) * tf.square(self.preds)
            else:
                local_losses += self.weight * ((1 - self.labels) / dual_scores) * tf.square(self.preds)
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-1000, clip_value_max=1000)
            numerator = tf.reduce_sum(self.labels + self.weight * (1 - self.labels))
            self.unbiased_loss = tf.reduce_sum(local_losses) / numerator
            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.unbiased_loss + self.lam * reg_embeds

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)


@dataclass
class PointwiseRecommender_ours(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""
    num_users: np.array
    num_items: np.array
    dim: int
    lam1: float
    lam2: float    
    lam3: float    
    eta: float
    weight: float = 1.
    clip: float = 0.
    dual_unbias: bool = False
    pow: float = 0.5

    def __post_init__(self, ) -> None:
        """Initialize Class."""
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph')
        self.items = tf.placeholder(tf.int32, [None], name='item_ph')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_ph')

        self.users_pair = tf.placeholder(tf.int32, [None], name='user_ph1')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_ph1')
        self.items2 = tf.placeholder(tf.int32, [None], name='item_ph2')
        self.point_R = tf.placeholder(tf.float32, [None, 1], name='point_R')
        self.point_O = tf.placeholder(tf.float32, [None, 1], name='point_O')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=42))
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=42))
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)

            self.u_embed_pair = tf.nn.embedding_lookup(self.user_embeddings, self.users_pair)

            self.i_p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            self.i_embed2 = tf.nn.embedding_lookup(self.item_embeddings, self.items2)
            
            self.representation = tf.concat([self.u_embed, self.i_embed], axis=0)

        with tf.variable_scope('prediction'):
            self.preds1 = tf.reduce_sum(tf.multiply(self.u_embed_pair, self.i_p_embed), 1)
            self.preds2 = tf.reduce_sum(tf.multiply(self.u_embed_pair, self.i_embed2), 1)
            self.preds_pair = tf.sigmoid(tf.expand_dims(self.preds1 - self.preds2, 1))

            self.logits = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.sigmoid(tf.expand_dims(self.logits, 1), name='sigmoid_prediction')

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the unbiased loss for the ideal loss function with binary implicit feedback.
            scores = tf.clip_by_value(self.scores, clip_value_min=self.clip, clip_value_max=1.0)
            orig_scores = tf.pow(scores, 1.0 / self.pow)
            dual_scores = tf.pow(1.0 - orig_scores, self.pow) + 1e-6
            local_losses = (self.labels / scores) * tf.square(1. - self.preds)
            if not self.dual_unbias:
                local_losses += self.weight * (1 - self.labels / scores) * tf.square(self.preds)
            else:
                local_losses += self.weight * ((1 - self.labels) / dual_scores) * tf.square(self.preds)
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-1000, clip_value_max=1000)
            numerator = tf.reduce_sum(self.labels + self.weight * (1 - self.labels))
            self.unbiased_loss = tf.reduce_sum(local_losses) / numerator

            self.pair_loss = -tf.log(self.preds)

            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)

            R1_indices = tf.where(tf.equal(self.point_R, 1))[:, 0]
            
            # R1 = tf.gather(self.R_labels, R1_indices)

            R1_embeddings = tf.gather(self.representation, R1_indices)
            # R0_embeddings = tf.gather(self.representation, R0_indices)
            O_labels = tf.gather(self.point_O, R1_indices) 
            R1O1 = tf.cast(tf.equal(O_labels, 1), tf.int32)
            
            # IPM Loss 1: R1 和 R0 之间的 Wasserstein 距离
            ###############################################################################################
            
            ipm_loss_ro1, _ = self.wasserstein(X=self.representation, t=tf.cast(tf.reshape(self.point_R, [-1]), tf.int32), p=0.5)
            ipm_loss_r1_r0, _ = self.wasserstein(X=R1_embeddings, t=R1O1, p=0.5)
            self.ipm_loss = ipm_loss_ro1 + ipm_loss_r1_r0

            self.loss = self.unbiased_loss + self.lam1 * reg_embeds + self.lam3 * self.pair_loss #+ self.lam2 * self.ipm_loss

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)


    SQRT_CONST = 1e-10

    def safe_sqrt(self, x, lbound=SQRT_CONST):
        ''' Numerically safe version of TensorFlow sqrt '''
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

    def pdist2sq(self, X, Y):
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2 * tf.matmul(X, tf.transpose(Y))
        nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
        D = (C + tf.transpose(ny)) + nx
        return D

    def pdist2(self, X, Y):
        """ Returns the tensorflow pairwise distance matrix """
        return self.safe_sqrt(self.pdist2sq(X, Y))

    def wasserstein(self, X, t, p, lam=10, its=10, sq=False, backpropT=False):
        """
        计算不同处理组之间的 Wasserstein 距离

        参数:
            X: [batch, embedding] 的嵌入矩阵
            t: [batch] 的处理标签，0 或 1
            p: 概率分布
            lam: 正则化强度
            its: Sinkhorn 迭代次数
            sq: 是否返回平方的 Wasserstein 距离
            backpropT: 是否允许 T 的反向传播

        返回:
            Wasserstein 距离 D 和正则化的距离矩阵 Mlam
        """
        p = tf.constant(p, dtype=tf.float32)

        # 获取 t == 0 和 t == 1 的索引
        it = tf.where(tf.equal(t, 1))[:, 0]
        ic = tf.where(tf.equal(t, 0))[:, 0]

        # 根据索引获取相应的 X
        Xc = tf.gather(X, ic)
        Xt = tf.gather(X, it)

        # 获取组的大小
        nc = tf.to_float(tf.shape(Xc)[0])
        nt = tf.to_float(tf.shape(Xt)[0])

        # print(nc.shape, nt.shape)
        # if len(nc.shape) == 0 or len(nt.shape) == 0:
            # return tf.constant(0.0), tf.constant(0.0)
        # 计算距离矩阵
        M = self.pdist2sq(Xt, Xc) if sq else self.safe_sqrt(self.pdist2sq(Xt, Xc))  # 动态维度兼容
        # 估计 lambda 和 delta
        M_mean = tf.reduce_mean(M)  # 动态矩阵均值
        delta = tf.stop_gradient(tf.reduce_max(M))  # 停止 delta 的梯度传播
        eff_lam = tf.stop_gradient(lam / M_mean)  # 计算有效的 lambda

        # 填充矩阵，添加一行一列
        Mt = M
        # row = delta*tf.ones(tf.shape(M[0:1,:]))
        # col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
        # Mt = tf.concat([M,row],0)
        # Mt = tf.concat([Mt,col],1)


        
        # Get dynamic shapes within the graph
        rows = tf.shape(M)[0]
        cols = tf.shape(M)[1]

        # Pad the matrix
        Mt = tf.pad(M, paddings=[[0, 1], [0, 1]], mode='CONSTANT', 
                    constant_values=delta)
        # Update the bottom-right corner using dynamic indices
        Mt = tf.tensor_scatter_nd_update(Mt, [[rows, cols]], [0.0])


        # 构造边缘向量，确保统一使用 float32
        a = tf.pad(p*tf.ones(tf.shape(tf.where(tf.equal(t, 1))[:,0:1]))/nt, paddings=[[0, 1], [0, 0]], mode='CONSTANT', constant_values=1-p)
        b = tf.pad((1-p)*tf.ones(tf.shape(tf.where(tf.equal(t, 0))[:,0:1]))/nc, paddings=[[0, 1], [0, 0]], mode='CONSTANT', constant_values=p)

        # a = tf.expand_dims(a, 1) # 变成二维张量
        # b = tf.expand_dims(b, 1) # 变成二维张量

        # 计算核矩阵
        # 计算核矩阵
        
        Mlam = eff_lam*Mt
        K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
        U = K*Mt
        ainvK = K/a

        u = a
        for i in range(0,its):
            u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
        v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

        T = u*(tf.transpose(v)*K)

        if not backpropT:
            T = tf.stop_gradient(T)

        E = T*Mt
        D = 2*tf.reduce_sum(E)

        return D, Mlam            


@dataclass
class PairwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pairwise approach."""
    num_users: np.array
    num_items: np.array
    dim: int = 20
    lam: float = 1e-4
    eta: float = 0.001
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph1')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_ph1')
        self.scores1 = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.items2 = tf.placeholder(tf.int32, [None], name='item_ph2')
        self.scores2 = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels2 = tf.placeholder(tf.float32, [None, 1], name='label_ph2')
        self.rel1 = tf.placeholder(tf.float32, [None, 1], name='rel_ph1')
        self.rel2 = tf.placeholder(tf.float32, [None, 1], name='rel_ph2')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            self.i_embed2 = tf.nn.embedding_lookup(self.item_embeddings, self.items2)

        with tf.variable_scope('prediction'):
            self.preds1 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_p_embed), 1)
            self.preds2 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed2), 1)
            self.preds = tf.sigmoid(tf.expand_dims(self.preds1 - self.preds2, 1))
            # print(self.preds)            
    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the naive pairwise loss.
            local_losses = - self.rel1 * (1 - self.rel2) * tf.log(self.preds)
            self.ideal_loss = tf.reduce_sum(local_losses) / tf.reduce_sum(self.rel1 * (1 - self.rel2))
            # define the unbiased pairwise loss.
            local_losses = - (1 / self.scores1) * (1 - (self.labels2 / self.scores2)) * tf.log(self.preds)
            # non-negative
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-self.beta, clip_value_max=10e5)
            self.unbiased_loss = tf.reduce_mean(local_losses)
            
            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)

            self.loss = self.unbiased_loss + self.lam * reg_embeds
            # print(self.unbiased_loss)
            # exit()
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)





@dataclass
class PairwiseRecommender_ours(AbstractRecommender):
    """Implicit Recommenders based on pairwise approach."""
    num_users: np.array
    num_items: np.array
    dim: int = 20
    lam1: float = 1e-4
    lam2: float = 1e-4
    lam3: float = 1e-4
    weight: float = 0.5
    ind: int = 0
    eta: float = 0.001
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph1')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_ph1')
        self.scores1 = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.items2 = tf.placeholder(tf.int32, [None], name='item_ph2')
        self.point_users = tf.placeholder(tf.int32, [None], name='point_user_ph')
        self.point_items = tf.placeholder(tf.int32, [None], name='point_item_ph')
        self.point_labels = tf.placeholder(tf.float32, [None, 1], name='point_label_ph')
        self.scores2 = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels2 = tf.placeholder(tf.float32, [None, 1], name='label_ph2')
        self.rel1 = tf.placeholder(tf.float32, [None, 1], name='rel_ph1')
        self.rel2 = tf.placeholder(tf.float32, [None, 1], name='rel_ph2')
        self.point_R = tf.placeholder(tf.float32, [None, 1], name='point_R')
        self.point_O = tf.placeholder(tf.float32, [None, 1], name='point_O')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.u_embed_point = tf.nn.embedding_lookup(self.user_embeddings, self.point_users)

            self.i_p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            self.i_embed2 = tf.nn.embedding_lookup(self.item_embeddings, self.items2)

            self.point_i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.point_items)
            
            self.representation = tf.concat([self.u_embed_point, self.point_i_embed], axis=0)

        with tf.variable_scope('prediction'):
            self.preds1 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_p_embed), 1)
            self.preds2 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed2), 1)
            self.preds = tf.sigmoid(tf.expand_dims(self.preds1 - self.preds2, 1))

            self.preds_point = tf.sigmoid(tf.expand_dims(tf.reduce_sum(tf.multiply(self.u_embed_point, self.point_i_embed), 1), 1))

            # self.preds2_point = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed2), 1)
            # print(self.preds)            
    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the naive pairwise loss.
            local_losses = - self.rel1 * (1 - self.rel2) * tf.log(self.preds)
            self.ideal_loss = tf.reduce_sum(local_losses) / tf.reduce_sum(self.rel1 * (1 - self.rel2))
            # define the unbiased pairwise loss.
            local_losses = - (1 / self.scores1) * (1 - (self.labels2 / self.scores2)) * tf.log(self.preds)
            # non-negative
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-self.beta, clip_value_max=10e5)
            self.unbiased_loss = tf.reduce_mean(local_losses)
            
            # self.point_loss = binary_cross_entropy(self.point_labels, self.preds_point, self.ind)
            weights = tf.where(tf.equal(self.point_labels, 1), tf.ones_like(self.point_labels), self.weight * tf.ones_like(self.point_labels))
            self.point_loss = tf.reduce_mean(weights * binary_cross_entropy(self.point_labels, self.preds_point, self.ind))
            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)

            # self.loss = self.unbiased_loss + self.lam2 * self.point_loss + self.lam1 * reg_embeds
            # print('reg_embeds', reg_embeds) 1000开始，40个epoch validation变成50，point 4，pairwise 0.6-1
            self.reg_loss = reg_embeds
            # self.loss = self.point_loss + self.lam1 * self.reg_loss            
            # print(self.unbiased_loss)
            # exit()

            R1_indices = tf.where(tf.equal(self.point_R, 1))[:, 0]

            # R1 = tf.gather(self.R_labels, R1_indices)

            R1_embeddings = tf.gather(self.representation, R1_indices)
            # R0_embeddings = tf.gather(self.representation, R0_indices)
            O_labels = tf.gather(self.point_O, R1_indices) 
            R1O1 = tf.cast(tf.equal(O_labels, 1), tf.int32)
            
            # IPM Loss 1: R1 和 R0 之间的 Wasserstein 距离
            ###############################################################################################
            
            ipm_loss_ro1, _ = self.wasserstein(X=self.representation, t=tf.cast(tf.reshape(self.point_R, [-1]), tf.int32), p=0.5)
            ipm_loss_r1_r0, _ = self.wasserstein(X=R1_embeddings, t=R1O1, p=0.5)
            self.ipm_loss = ipm_loss_ro1 + ipm_loss_r1_r0
            
            # print('1', self.ipm_loss)
            self.loss = self.point_loss + self.lam2 * self.unbiased_loss + self.lam1 * reg_embeds + self.lam3 * self.ipm_loss

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)

    SQRT_CONST = 1e-10

    def safe_sqrt(self, x, lbound=SQRT_CONST):
        ''' Numerically safe version of TensorFlow sqrt '''
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

    def pdist2sq(self, X, Y):
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2 * tf.matmul(X, tf.transpose(Y))
        nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
        D = (C + tf.transpose(ny)) + nx
        return D

    def pdist2(self, X, Y):
        """ Returns the tensorflow pairwise distance matrix """
        return self.safe_sqrt(self.pdist2sq(X, Y))

    def wasserstein(self, X, t, p, lam=10, its=10, sq=False, backpropT=False):
        """
        计算不同处理组之间的 Wasserstein 距离

        参数:
            X: [batch, embedding] 的嵌入矩阵
            t: [batch] 的处理标签，0 或 1
            p: 概率分布
            lam: 正则化强度
            its: Sinkhorn 迭代次数
            sq: 是否返回平方的 Wasserstein 距离
            backpropT: 是否允许 T 的反向传播

        返回:
            Wasserstein 距离 D 和正则化的距离矩阵 Mlam
        """
        p = tf.constant(p, dtype=tf.float32)

        # 获取 t == 0 和 t == 1 的索引
        it = tf.where(tf.equal(t, 1))[:, 0]
        ic = tf.where(tf.equal(t, 0))[:, 0]

        # 根据索引获取相应的 X
        Xc = tf.gather(X, ic)
        Xt = tf.gather(X, it)

        # 获取组的大小
        nc = tf.to_float(tf.shape(Xc)[0])
        nt = tf.to_float(tf.shape(Xt)[0])

        # print(nc.shape, nt.shape)
        # if len(nc.shape) == 0 or len(nt.shape) == 0:
            # return tf.constant(0.0), tf.constant(0.0)
        # 计算距离矩阵
        M = self.pdist2sq(Xt, Xc) if sq else self.safe_sqrt(self.pdist2sq(Xt, Xc))  # 动态维度兼容
        # 估计 lambda 和 delta
        M_mean = tf.reduce_mean(M)  # 动态矩阵均值
        delta = tf.stop_gradient(tf.reduce_max(M))  # 停止 delta 的梯度传播
        eff_lam = tf.stop_gradient(lam / M_mean)  # 计算有效的 lambda

        # 填充矩阵，添加一行一列
        Mt = M
        # row = delta*tf.ones(tf.shape(M[0:1,:]))
        # col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
        # Mt = tf.concat([M,row],0)
        # Mt = tf.concat([Mt,col],1)


        
        # Get dynamic shapes within the graph
        rows = tf.shape(M)[0]
        cols = tf.shape(M)[1]

        # Pad the matrix
        Mt = tf.pad(M, paddings=[[0, 1], [0, 1]], mode='CONSTANT', 
                    constant_values=delta)
        # Update the bottom-right corner using dynamic indices
        Mt = tf.tensor_scatter_nd_update(Mt, [[rows, cols]], [0.0])


        # 构造边缘向量，确保统一使用 float32
        a = tf.pad(p*tf.ones(tf.shape(tf.where(tf.equal(t, 1))[:,0:1]))/nt, paddings=[[0, 1], [0, 0]], mode='CONSTANT', constant_values=1-p)
        b = tf.pad((1-p)*tf.ones(tf.shape(tf.where(tf.equal(t, 0))[:,0:1]))/nc, paddings=[[0, 1], [0, 0]], mode='CONSTANT', constant_values=p)

        # a = tf.expand_dims(a, 1) # 变成二维张量
        # b = tf.expand_dims(b, 1) # 变成二维张量

        # 计算核矩阵
        # 计算核矩阵
        
        Mlam = eff_lam*Mt
        K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
        U = K*Mt
        ainvK = K/a

        u = a
        for i in range(0,its):
            u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
        v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

        T = u*(tf.transpose(v)*K)

        if not backpropT:
            T = tf.stop_gradient(T)

        E = T*Mt
        D = 2*tf.reduce_sum(E)

        return D, Mlam

@dataclass
class UPLPairwiseRecommender(PairwiseRecommender):
    """Implicit Recommenders based on pairwise approach."""
    pair_weight: int = 0
    norm_weight: bool = False

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.user_b = tf.Variable(tf.random_normal(shape=[self.num_users], stddev=0.01), name='user_b')
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.item_b = tf.Variable(tf.random_normal(shape=[self.num_items], stddev=0.01), name='item_b')

            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.u_bias = tf.nn.embedding_lookup(self.user_b, self.users)
            self.i_p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            self.i_p_bias = tf.nn.embedding_lookup(self.item_b, self.pos_items)

            self.i_embed2 = tf.nn.embedding_lookup(self.item_embeddings, self.items2)
            self.i_bias2 = tf.nn.embedding_lookup(self.item_b, self.items2)

        with tf.variable_scope('prediction'):
            self.preds1 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_p_embed), 1)  # + self.u_bias + self.i_p_bias
            self.preds2 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed2), 1)  # + self.u_bias + self.i_bias2
            self.preds = tf.sigmoid(tf.expand_dims(self.preds1 - self.preds2, 1))

        with tf.name_scope('point_embedding_layer'):
            self.point_user_embeddings = tf.get_variable('point_user_embeddings', shape=[self.num_users, self.dim],
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.point_user_b = tf.Variable(tf.random_normal(shape=[self.num_users], stddev=0.01), name='point_user_b')
            self.point_item_embeddings = tf.get_variable('point_item_embeddings', shape=[self.num_items, self.dim],
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.point_item_b = tf.Variable(tf.random_normal(shape=[self.num_items], stddev=0.01), name='point_item_b')
            self.point_global_bias = tf.get_variable('point_global_bias', [1],
                                                     initializer=tf.constant_initializer(1e-3, dtype=tf.float32))

            self.point_u_embed = tf.nn.embedding_lookup(self.point_user_embeddings, self.users)
            self.point_u_bias = tf.nn.embedding_lookup(self.point_user_b, self.users)
            self.point_i_embed = tf.nn.embedding_lookup(self.point_item_embeddings, self.items2)
            self.point_i_bias = tf.nn.embedding_lookup(self.point_item_b, self.items2)
            self.point_pos_i_embed = tf.nn.embedding_lookup(self.point_item_embeddings, self.pos_items)
            self.point_pos_i_bias = tf.nn.embedding_lookup(self.point_item_b, self.pos_items)

        with tf.variable_scope('point_prediction'):
            self.point_logits = tf.reduce_sum(tf.multiply(self.point_u_embed, self.point_i_embed), 1)
            self.point_logits = tf.add(self.point_logits, self.point_u_bias)
            self.point_logits = tf.add(self.point_logits, self.point_i_bias)
            self.point_logits = tf.add(self.point_logits, self.point_global_bias)
            self.point_preds = tf.sigmoid(tf.expand_dims(self.point_logits, 1), name='point_preds')

            self.point_pos_logits = tf.reduce_sum(tf.multiply(self.point_u_embed, self.point_pos_i_embed), 1)
            self.point_pos_logits = tf.add(self.point_pos_logits, self.point_u_bias)
            self.point_pos_logits = tf.add(self.point_pos_logits, self.point_pos_i_bias)
            self.point_pos_logits = tf.add(self.point_pos_logits, self.point_global_bias)
            self.point_pos_preds = tf.sigmoid(tf.expand_dims(self.point_pos_logits, 1), name='point_pos_preds')

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the naive pairwise loss.
            local_losses = - self.rel1 * (1 - self.rel2) * tf.log(self.preds)
            self.ideal_loss = tf.reduce_sum(local_losses) / tf.reduce_sum(self.rel1 * (1 - self.rel2))
            # define the unbiased pairwise loss.
            weight = (1 / self.scores1) * ((1 - self.labels2) / self.scores2)
            local_losses = -(weight * tf.log(self.preds))
            point_pos_preds = tf.stop_gradient(self.point_pos_preds)
            point_neg_preds = tf.stop_gradient(self.point_preds)
            numerator = (self.scores2 * (1 - point_neg_preds))
            denominator = 1.0 - point_neg_preds * self.scores2 + 1e-5
            self.scores2_minus = numerator / denominator
            local_losses *= self.scores2_minus
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-self.beta, clip_value_max=10e5)
            self.unbiased_loss = tf.reduce_mean(local_losses)

            local_ce = (self.labels2 / self.scores2) * tf.log(self.point_preds)
            local_ce += (1 - self.labels2 / self.scores2) * tf.log(1. - self.point_preds)
            self.weighted_ce = - tf.reduce_mean(local_ce)

            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)
            reg_embeds += tf.nn.l2_loss(self.user_b)
            reg_embeds += tf.nn.l2_loss(self.item_b)
            reg_embeds += tf.nn.l2_loss(self.point_user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.point_item_embeddings)
            reg_embeds += tf.nn.l2_loss(self.point_user_b)
            reg_embeds += tf.nn.l2_loss(self.point_item_b)
            self.loss = self.unbiased_loss + self.lam * reg_embeds + self.weighted_ce