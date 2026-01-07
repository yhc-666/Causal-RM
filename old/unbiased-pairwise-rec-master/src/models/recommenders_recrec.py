from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
from io import SEEK_CUR

import numpy as np
from numpy.core.fromnumeric import clip
import tensorflow as tf

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


class ReCRec_I(AbstractRecommender):
    
    def __init__(self, num_users: np.array, num_items: np.array,pscore:np.ndarray,
                 dim: int, lam: float, eta: float) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.pscore = pscore 

        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_placeholder')
        self.exposure = tf.placeholder(tf.float32, [None, 1], name='exposure_placeholder')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_placeholder')
        

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings = tf.get_variable(
                'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(
                'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            
            
            self.user_bias_rel = tf.get_variable(
                'user_bias_rel', shape=[self.num_users, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_bias_rel = tf.get_variable(
                'item_bias_rel', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            
            self.item_mu = tf.get_variable(
                'item_mu', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            
            # lookup embeddings of current batch
            self.u_embed = tf.nn.embedding_lookup(
                self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(
                self.item_embeddings, self.items)
            self.u_bias_rel = tf.nn.embedding_lookup(
                self.user_bias_rel, self.users)
            self.i_bias_rel = tf.nn.embedding_lookup(
                self.item_bias_rel, self.items)
            
            self.i_mu = tf.nn.embedding_lookup(
                self.item_mu, self.items)
            
        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(
                tf.multiply(self.u_embed, self.i_embed), 1)
            self.gama = tf.sigmoid(tf.expand_dims(
                self.logits, 1)+self.u_bias_rel+self.i_bias_rel, name='gama_prediction')
            self.mu = tf.sigmoid(self.i_mu, name='mu_prediction')
           
            
            
    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            
            p = self.mu *(1-self.gama)/(1-self.mu*self.gama)
            q = self.gama *(1-self.mu)/(1-self.mu*self.gama)
            
            p = (1-self.labels)*p + self.labels
            q = (1-self.labels)*q + self.labels

            p = tf.stop_gradient(p)
            q = tf.stop_gradient(q)
            
            self.ce_mu = -tf.reduce_mean((p) * tf.log( self.mu) +(1 - p) * tf.log(1-self.mu))
            self.ce_gama = -tf.reduce_mean((q ) * tf.log(self.gama) +(1 - q) * tf.log(1-self.gama))            
            self.ce_label = -tf.reduce_mean((self.labels ) * tf.log(self.gama*self.mu) +(1 - self.labels) * tf.log(1-self.gama*self.mu))

            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)

            reg_term_embeds2 = tf.nn.l2_loss(self.item_mu)

            self.loss_mu = self.ce_mu + self.lam *10*reg_term_embeds2 
            self.loss_gama = self.ce_gama + self.lam * reg_term_embeds +self.ce_label

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads_mu = tf.train.AdamOptimizer(
               learning_rate=self.eta).minimize(self.loss_mu,var_list=[self.item_mu])
            self.apply_grads_gama = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss_gama,var_list=[self.item_embeddings,self.user_embeddings,self.item_bias_rel,self.user_bias_rel])

class ReCRec_D(AbstractRecommender):
    #share  some dims
    def __init__(self, num_users: np.array, num_items: np.array,pscore:np.ndarray,
                 dim: int, lam: float, eta: float, dim2:int = 100) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.dim2 = dim2
        self.lam = lam
        self.eta = eta
        self.pscore = pscore

        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(
            tf.float32, [None, 1], name='score_placeholder')
        self.labels = tf.placeholder(
            tf.float32, [None, 1], name='label_placeholder')
        self.exposure = tf.placeholder(
            tf.float32, [None, 1], name='exposure_placeholder')

    def build_graph(self) -> None:
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings_expo = tf.get_variable(
                'user_embeddings_expo', shape=[self.num_users,self.dim2],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings_expo = tf.get_variable(
                'item_embeddings_expo', shape=[self.num_items, self.dim2],
                initializer=tf.contrib.layers.xavier_initializer())
            
            
            self.user_embeddings_rel = tf.get_variable(
                'user_embeddings_rel', shape=[self.num_users, self.dim2],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings_rel = tf.get_variable(
                'item_embeddings_rel', shape=[self.num_items, self.dim2],
                initializer=tf.contrib.layers.xavier_initializer())
            
            self.user_embeddings_shared = tf.get_variable(
                'user_embeddings_shared', shape=[self.num_users, self.dim-self.dim2],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings_shared = tf.get_variable(
                'item_embeddings_shared', shape=[self.num_items, self.dim-self.dim2],
                initializer=tf.contrib.layers.xavier_initializer())
            
            self.user_bias_expo = tf.get_variable(
                'user_bias_expo', shape=[self.num_users, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_bias_expo = tf.get_variable(
                'item_bias_expo', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            

            self.user_bias_rel = tf.get_variable(
                'user_bias_rel', shape=[self.num_users, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_bias_rel = tf.get_variable(
                'item_bias_rel', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            
            # lookup embeddings of current batch
            self.u_embed_expo = tf.nn.embedding_lookup(
                self.user_embeddings_expo, self.users)
            self.i_embed_expo = tf.nn.embedding_lookup(
                self.item_embeddings_expo, self.items)

            self.u_embed_rel = tf.nn.embedding_lookup(
                self.user_embeddings_rel, self.users)
            self.i_embed_rel = tf.nn.embedding_lookup(
                self.item_embeddings_rel, self.items)

            
            self.u_embed_shared = tf.nn.embedding_lookup(
                self.user_embeddings_shared, self.users)
            self.i_embed_shared = tf.nn.embedding_lookup(
                self.item_embeddings_shared, self.items)


            self.u_bias_expo = tf.nn.embedding_lookup(
                self.user_bias_expo, self.users)
            self.i_bias_expo = tf.nn.embedding_lookup(
                self.item_bias_expo, self.items)

            self.u_bias_rel = tf.nn.embedding_lookup(
                self.user_bias_rel, self.users)
            self.i_bias_rel = tf.nn.embedding_lookup(
                self.item_bias_rel, self.items)

            

            

        with tf.variable_scope('prediction'):
            
            self.logits_expo = tf.reduce_sum(
                tf.multiply(self.u_embed_expo, self.i_embed_expo), 1)
            
            self.logits_rel = tf.reduce_sum(
                tf.multiply(self.u_embed_rel, self.i_embed_rel), 1)
            
            
            self.logits_shared = tf.reduce_sum(
                tf.multiply(self.u_embed_shared, self.i_embed_shared), 1)
            
            
            self.user_embeddings = tf.concat([self.user_embeddings_rel,self.user_embeddings_shared],1)
            self.item_embeddings = tf.concat([self.item_embeddings_rel,self.item_embeddings_shared],1)
            
            self.gama = tf.sigmoid(tf.expand_dims(tf.add(self.logits_rel,self.logits_shared), 1)+self.u_bias_rel+self.i_bias_rel, name='gama_prediction')
            
            self.mu = tf.sigmoid(tf.expand_dims(tf.add(self.logits_expo,self.logits_shared), 1)+self.u_bias_expo+self.i_bias_expo, name='mu_prediction')
            
            
            
    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            p = self.mu *(1-self.gama)/(1-self.mu*self.gama)
            q = self.gama *(1-self.mu)/(1-self.mu*self.gama)
            
            
            p = (1-self.exposure)*p + self.exposure
            
            q = (1-self.labels)*(1-self.exposure)*q + self.labels
            
            p = tf.stop_gradient(p)
            q = tf.stop_gradient(q)

            self.ce_mu = -tf.reduce_mean((p) * tf.log( self.mu) +(1 - p) * tf.log(1-self.mu))
            self.ce_gama = -tf.reduce_mean((q ) * tf.log(self.gama) +(1 - q) * tf.log(1-self.gama))
            self.ce_label = -tf.reduce_mean((self.labels ) * tf.log(self.gama*self.mu) +(1 - self.labels) * tf.log(1-self.gama*self.mu))

            reg_term_embeds_rel = tf.nn.l2_loss(self.user_embeddings_rel) \
                + tf.nn.l2_loss(self.item_embeddings_rel)

            reg_term_embeds_shared = tf.nn.l2_loss(self.user_embeddings_shared) \
                + tf.nn.l2_loss(self.item_embeddings_shared)
            reg_term_embeds_expo = tf.nn.l2_loss(self.user_embeddings_expo) \
                + tf.nn.l2_loss(self.item_embeddings_expo)
            
            
            pscore_loss = tf.reduce_mean(tf.square(self.mu-self.scores))
            
            self.loss_mu = self.ce_mu + self.lam *100*(reg_term_embeds_expo+reg_term_embeds_shared) +pscore_loss
            self.loss_gama = self.ce_gama + self.lam * (reg_term_embeds_rel+reg_term_embeds_shared) +self.ce_label


    def add_optimizer(self) -> None:
        with tf.name_scope('optimizer'):
            self.apply_grads_mu = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss_mu,var_list=[self.item_embeddings_expo,self.user_embeddings_expo,self.user_bias_expo,self.item_bias_expo,self.item_embeddings_shared,self.user_embeddings_shared])
            self.apply_grads_gama = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss_gama,var_list=[self.item_embeddings_rel,self.user_embeddings_rel,self.item_embeddings_shared,self.user_embeddings_shared,self.item_bias_rel,self.user_bias_rel])

class ReCRec_F(AbstractRecommender):
    # with MF model with biasi + bias u
    def __init__(self, num_users: np.array, num_items: np.array,pscore:np.ndarray,
                 dim: int, lam: float,lamp: float ,eta: float) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.lamp = lamp
        self.eta = eta
        self.pscore = pscore
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(
            tf.float32, [None, 1], name='score_placeholder')
        self.labels = tf.placeholder(
            tf.float32, [None, 1], name='label_placeholder')
        self.exposure = tf.placeholder(tf.float32, [None, 1], name='exposure_placeholder')
        

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings_expo = tf.get_variable(
                'user_embeddings_expo', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings_expo = tf.get_variable(
                'item_embeddings_expo', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            
            self.user_bias_expo = tf.get_variable(
                'user_bias_expo', shape=[self.num_users, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_bias_expo = tf.get_variable(
                'item_bias_expo', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            

            self.user_bias_rel = tf.get_variable(
                'user_bias_rel', shape=[self.num_users, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_bias_rel = tf.get_variable(
                'item_bias_rel', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            self.user_embeddings = tf.get_variable(
                'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(
                'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            
            
            # lookup embeddings of current batch
            self.u_embed_expo = tf.nn.embedding_lookup(
                self.user_embeddings_expo, self.users)
            self.i_embed_expo = tf.nn.embedding_lookup(
                self.item_embeddings_expo, self.items)

            self.u_bias_expo = tf.nn.embedding_lookup(
                self.user_bias_expo, self.users)
            self.i_bias_expo = tf.nn.embedding_lookup(
                self.item_bias_expo, self.items)

            self.u_embed_rel = tf.nn.embedding_lookup(
                self.user_embeddings, self.users)
            self.i_embed_rel = tf.nn.embedding_lookup(
                self.item_embeddings, self.items)

            self.u_bias_rel = tf.nn.embedding_lookup(
                self.user_bias_rel, self.users)
            self.i_bias_rel = tf.nn.embedding_lookup(
                self.item_bias_rel, self.items)

            
            

        with tf.variable_scope('prediction'):
            self.logits_expo = tf.reduce_sum(
                tf.multiply(self.u_embed_expo, self.i_embed_expo), 1)
            
            self.logits_rel = tf.reduce_sum(
                tf.multiply(self.u_embed_rel, self.i_embed_rel), 1)
            
            
            self.gama = tf.sigmoid(tf.expand_dims(
                self.logits_rel, 1)+self.u_bias_rel+self.i_bias_rel, name='gama_prediction')
            
            self.mu = tf.sigmoid(tf.expand_dims(
                self.logits_expo, 1)+self.u_bias_expo+self.i_bias_expo, name='mu_prediction')
            
            
            
            
    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            
            p = self.mu *(1-self.gama)/(1-self.mu*self.gama)
            q = self.gama *(1-self.mu)/(1-self.mu*self.gama)
            
            p = (1-self.labels)*p + self.labels
            q = (1-self.labels)*q + self.labels
    
            p = tf.stop_gradient(p)
            q = tf.stop_gradient(q)

            
            self.ce_mu = -tf.reduce_mean((p) * tf.log( self.mu) +(1 - p) * tf.log(1-self.mu))
            self.ce_gama = -tf.reduce_mean((q ) * tf.log(self.gama) +(1 - q) * tf.log(1-self.gama))
            self.ce_label = -tf.reduce_mean((self.labels ) * tf.log(self.gama*self.mu) +(1 - self.labels) * tf.log(1-self.gama*self.mu))
            
            reg_term_embeds_rel = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings) 

            reg_term_embeds_expo = tf.nn.l2_loss(self.user_embeddings_expo) \
                + tf.nn.l2_loss(self.item_embeddings_expo) 
            
            
            pscore_loss = tf.reduce_mean(tf.square(self.mu-self.scores))
            
            self.loss_mu = self.ce_mu + self.lam *50*reg_term_embeds_expo +self.lamp*pscore_loss
            self.loss_gama = self.ce_gama + self.lam * reg_term_embeds_rel +self.ce_label


    def add_optimizer(self) -> None:
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads_mu = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss_mu,var_list=[self.item_embeddings_expo,self.user_embeddings_expo,self.user_bias_expo,self.item_bias_expo])
            self.apply_grads_gama = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss_gama,var_list=[self.item_embeddings,self.user_embeddings,self.item_bias_rel,self.user_bias_rel])


