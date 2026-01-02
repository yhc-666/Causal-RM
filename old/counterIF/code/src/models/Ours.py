import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息

import tensorflow as tf
import numpy as np
from sklearn.svm import OneClassSVM
from tensorflow.python import debug as tf_debug
from tensorflow.python import debug as tf_debug
from tqdm import tqdm


class Ours:
    def __init__(self, sess, train, val, num_users, num_items, hidden_dim, hidden_pre, embedding_dim, eta, max_iters,
                 batch_size, random_state, nu, percentile, HN_percentile, subsample, data_name, wd1, wd2, lambda1, lambda2):
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.eta = eta
        self.percentile=percentile
        self.HN_percentile=HN_percentile
        self.subsample = subsample
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.data_name = data_name
        self.wd1 = wd1
        self.wd2 = wd2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        
        self.train = train
        # self.train_pos_idx = np.where(self.train[:, 2] == 1)[0]
        self.val = val
        self.sess = sess # tf_debug.LocalCLIDebugWrapperSession(sess)
        self.random_state = random_state
        self.nu = nu
        self.hidden_pre = hidden_pre
        

        # self.train_ui_matrix = self.convert_to_matrix(train, num_users, num_items)
        # self.train_iu_matrix = np.copy(self.train_ui_matrix.T)

        self.create_placeholders()
        self.build_graph()
        self.compute_loss()
        self.add_optimizer()
        

    def add_optimizer(self):
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)  # 使用 AdamOptimizer
            

    def convert_to_matrix(self, data, num_users, num_items):
        """将输入数据转换为矩阵形式"""
        matrix = np.zeros((num_users, num_items))
        for (u, i, r) in data[:, :3]:
            matrix[int(u), int(i)] = r
        return matrix


    def create_placeholders(self):
        """创建输入数据、标签和表征的占位符"""
        with tf.name_scope("input_data"):
            # 用户和物品 ID 占位符
            self.user_input = tf.placeholder(tf.int32, [None], name='user_input')  # 用户 ID
            self.item_input = tf.placeholder(tf.int32, [None], name='item_input')  # 物品 ID

            # 标签占位符
            self.R_labels = tf.placeholder(tf.int32, [None], name="R_labels")  # R 标签
            self.O_labels = tf.placeholder(tf.int32, [None], name="O_labels")  # O 标签
            self.Y_labels = tf.placeholder(tf.int32, [None], name="Y_labels")  # Y 标签

    def build_graph(self):
        """构建 embedding、表征层和预测层"""
        with tf.name_scope("embedding_layer"):
            # Xavier 初始化器
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_state)

            # 用户和物品的 embedding
            self.user_embedding = tf.get_variable(
                name='user_embedding',
                shape=[self.num_users, self.embedding_dim],
                initializer=initializer,
                trainable=True
            )
            self.item_embedding = tf.get_variable(
                name='item_embedding',
                shape=[self.num_items, self.embedding_dim],
                initializer=initializer,
                trainable=True
            )


        with tf.name_scope("representation_layer"):
            # 使用 tf.nn.embedding_lookup 来提取用户和物品的嵌入
            user_embedded = tf.nn.embedding_lookup(self.user_embedding, self.user_input)
            item_embedded = tf.nn.embedding_lookup(self.item_embedding, self.item_input)

            # 拼接用户和物品嵌入
            self.concatenated_embedding = tf.concat([user_embedded, item_embedded], axis=1)

            # 构建多层线性表征层
            layer_output = self.concatenated_embedding
            for i, dim in enumerate(self.hidden_dim):
                layer_output = tf.layers.dense(layer_output, dim, activation=tf.nn.relu, name=f'hidden_layer_{i+1}')
            self.representation = layer_output

        with tf.name_scope("prediction_layer"):
            # 构建多层预测层（prediction hidden layers）
            prediction_output = self.representation
            for i, dim in enumerate(self.hidden_pre):
                prediction_output = tf.layers.dense(prediction_output, dim, activation=tf.nn.relu, name=f'prediction_hidden_layer_{i+1}')

            # 最终预测层
            self.prediction = tf.layers.dense(prediction_output, 1, activation=tf.sigmoid, name='final_prediction')
    def binary_cross_entropy(self, y_true, y_pred):
        """计算二分类交叉熵损失"""
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)  # 避免 log(0)
        return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    def sample_and_pair(self, positive_preds, negative_preds):
        """正负样本采样与配对"""
        num_pos = tf.shape(positive_preds)[0]
        num_neg = tf.shape(negative_preds)[0]
        num_samples = tf.minimum(num_pos, num_neg)

        # 停止梯度传播以避免反向传播中计算梯度
        pos_sample = positive_preds[:num_samples]
        neg_sample = negative_preds[:num_samples]

        pos_expanded = tf.expand_dims(pos_sample, axis=1)
        neg_expanded = tf.expand_dims(neg_sample, axis=0)

        pos_neg_pairs = pos_expanded - neg_expanded
        pos_neg_pairs_flat = tf.reshape(pos_neg_pairs, [-1])

        return pos_neg_pairs_flat

    def bpr_loss(self, positive_preds, negative_preds):
        """计算BPR损失"""
        pos_neg_pairs = self.sample_and_pair(positive_preds, negative_preds)
        return -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos_neg_pairs)))

    def align_bpr_loss(self):
        """仿照上面的bpr_loss函数，但是在采样时，优先保证同一个用户的正负样本对比学习"""
        
        """计算BPR损失，保证正负样本来自同一用户"""
        # 获取正负样本的掩码
        positive_mask = tf.cast(self.Y_labels, tf.bool)
        negative_mask = tf.logical_not(positive_mask)
        
        # 获取正样本的索引及对应的用户ID
        positive_indices = tf.reshape(tf.where(positive_mask), [-1])
        positive_users = tf.gather(self.user_input, positive_indices)
        
        # 获取负样本的索引及对应的用户ID
        negative_indices = tf.reshape(tf.where(negative_mask), [-1])
        negative_users = tf.gather(self.user_input, negative_indices)
        
        # 为每个正样本用户选取一个对应的负样本
        def get_negative_idx(u):
            # 找到同一用户的负样本索引
            user_neg_mask = tf.logical_and(negative_mask, tf.equal(self.user_input, u))
            user_neg_indices = tf.reshape(tf.where(user_neg_mask), [-1])
            # 如果user_neg_indices非空，随机选择一个负样本，否则从所有负样本negative_indices中随机选择一个
            return tf.cond(
                tf.greater(tf.shape(user_neg_indices)[0], 0),
                lambda: tf.random.shuffle(user_neg_indices)[0],
                lambda: tf.random.shuffle(negative_indices)[0]
            )
            
            # selected_idx = tf.random.shuffle(user_neg_indices)[0]
            # return selected_idx
        
        # 为所有正样本用户获取对应的负样本索引
        selected_negative_indices = tf.map_fn(
            get_negative_idx, positive_users, dtype=tf.int64
        )
        
        # 收集正负样本的预测值
        positive_preds = tf.gather(self.prediction, positive_indices)
        negative_preds = tf.gather(self.prediction, selected_negative_indices)
        
        # 计算BPR损失
        pos_neg_diff = positive_preds - negative_preds
        bpr_loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos_neg_diff)))
        
        return bpr_loss

        

      

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
        return safe_sqrt(pdist2sq(X, Y))
    def wasserstein(self, X, t, p, lam=10, its=10, sq=False, backpropT=False):
        """
        计算不同处理组之间的 Wasserstein 距离
        """
        p = tf.constant(p, dtype=tf.float32)

        # 获取 t == 0 和 t == 1 的索引
        it = tf.where(tf.equal(t, 0))[:, 0]
        ic = tf.where(tf.equal(t, 1))[:, 0]
        
        # 若任一组为空，返回零 loss 和一个与输入尺寸匹配的占位矩阵
        def empty_group():
            dummy_shape = tf.concat([tf.shape(X)[:-1], [tf.shape(X)[-1]+1]], axis=0)
            return tf.constant(0.0), tf.zeros(dummy_shape, dtype=tf.float32)
        
        non_empty = tf.logical_and(tf.greater(tf.size(it), 0), tf.greater(tf.size(ic), 0))
        return tf.cond(non_empty,
                    lambda: self._compute_wasserstein(X, t, p, lam, its, sq, backpropT, it, ic),
                    empty_group)
    def _compute_wasserstein(self, X, t, p, lam, its, sq, backpropT, it, ic):
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
        it = tf.where(tf.equal(t, 0))[:, 0]
        ic = tf.where(tf.equal(t, 1))[:, 0]

        # 根据索引获取相应的 X
        Xc = tf.gather(X, ic)
        Xt = tf.gather(X, it)

        # 获取组的大小
        nc = tf.to_float(tf.shape(Xc)[0])
        nt = tf.to_float(tf.shape(Xt)[0])

        # if len(nc.shape) == 0 or len(nt.shape) == 0:
        #     return 0,0
        # 计算距离矩阵
        M = self.pdist2sq(Xt, Xc) if sq else self.safe_sqrt(self.pdist2sq(Xt, Xc))  # 动态维度兼容

        # 估计 lambda 和 delta
        M_mean = tf.reduce_mean(M)  # 动态矩阵均值
        delta = tf.stop_gradient(tf.reduce_max(M))  # 停止 delta 的梯度传播
        eff_lam = tf.stop_gradient(lam / M_mean)  # 计算有效的 lambda

        # 填充矩阵，添加一行一列
        Mt = M
        row = delta*tf.ones(tf.shape(M[0:1,:]))
        col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
        Mt = tf.concat([M,row],0)
        Mt = tf.concat([Mt,col],1)

        # 构造边缘向量，确保统一使用 float32
        # a = tf.concat([p*tf.ones(tf.shape(tf.where(t==1)[:,0:1]))/nt, (1-p)*tf.ones((1,1))],0)
        # b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t==0)[:,0:1]))/nc, p*tf.ones((1,1))],0)

        a = tf.pad(p*tf.ones(tf.shape(tf.where(t==1)[:,0:1]))/nt, paddings=[[0, 1], [0, 0]], mode='CONSTANT', constant_values=1-p)
        b = tf.pad((1-p)*tf.ones(tf.shape(tf.where(t==0)[:,0:1]))/nc, paddings=[[0, 1], [0, 0]], mode='CONSTANT', constant_values=p)


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


    def adjust_nu_based_on_sample(self, user_emb, item_emb, subsample_size=10000, target_percentile=90, validate_y=True):
        """
        基于数据集的子样本调整 nu，而不是处理整个数据集。
        """
        # 获取完整的用户和物品嵌入
        train_pos_idx = np.where(self.train[:, 2] == 1)[0]
        # 如果整个数据集过大，随机选择一个子样本用于计算
        if len(train_pos_idx) > subsample_size:
            subsample_indices = np.random.choice(train_pos_idx, subsample_size, replace=False)
            subsample_data = self.train[subsample_indices]
        else:
            subsample_data = self.train[train_pos_idx]

        # 提取子样本中的用户和物品ID
        user_ids = subsample_data[:, 0].astype(np.int32)
        item_ids = subsample_data[:, 1].astype(np.int32)

        # 获取子样本的用户和物品嵌入
        user_emb_selected = user_emb[user_ids]
        item_emb_selected = item_emb[item_ids]

        # 拼接用户和物品的嵌入
        embeddings = np.concatenate([user_emb_selected, item_emb_selected], axis=1)

        # 使用 OneClassSVM 进行分类
        nu = self.nu
        while True:
            ocsvm = OneClassSVM(kernel='rbf', nu=nu)
            ocsvm.fit(embeddings)
            distances = ocsvm.decision_function(embeddings)

            # 获取标签为 Y=1 的样本
            # y_1_indices = np.where(subsample_data[:, 2] == 1)[0]
            # y_1_distances = distances[y_1_indices]

            # 调整 nu，确保目标百分位数的 Y=1 样本距离小于 0
            if validate_y:
                target_distance = np.percentile(distances, 100-target_percentile)
                if target_distance > 0:
                    break  # 目标已满足，跳出循环

            nu *= 0.95  # 每次减少 10%
        self.ocsvm = ocsvm
        return nu
    def assign_labels(self, distances, y_labels):
        """
        根据 SVM 分类结果和 Y 标签分配 R 和 O 标签.
        - Y == 1 的样本自动设定 R=1, O=1.
        - 其余样本根据 SVM 距离进行分类.
        """
        R_O_labels = []
        # 计算 y=0 的样本的 distances 10% 分位数
        y0_indices = np.where(y_labels == 0)[0]
        y0_distances = distances[y0_indices]
        distance_percentile = np.percentile(y0_distances, self.HN_percentile)#加超参数 10 20
        
        
        for i, distance in enumerate(distances):
            if y_labels[i] == 1:
                R_O_labels.append([1, 1])  # 如果 Y=1，直接设定 R=1, O=1
            else:
                # 其他样本根据 SVM 距离进行分类
                if distance > 0:  # 正的距离表示样本在超球体内部
                    R_O_labels.append([1, 1])  # R=1, O=1
                elif distance <= distance_percentile:  # 在超球体外部距离
                    R_O_labels.append([1, 0])  # R=1, O=0
                else:  # 在超球体外部
                    R_O_labels.append([0, -1])  # R=0, O=-1（或根据需求调整）

        return np.array(R_O_labels)

    def train_classifier(self, target_percentile=90, validate_y=True, subsample_size=10000):

        

        """
        使用子样本调整 nu 并训练分类器来生成 R 和 O 标签，Y=1 时直接设定 R=1, O=1.
        """
        # 先使用子样本调整 nu
        emb_path = f'../data/{self.data_name}_st_1/emb'
        print(emb_path)
        user_emb = np.load(f'{emb_path}/user_embed.npy')
        item_emb = np.load(f'{emb_path}/item_embed.npy')

        # self.nu = self.adjust_nu_based_on_sample(user_emb, item_emb, subsample_size=subsample_size, target_percentile=target_percentile,
        #                                              validate_y=validate_y)
        self.nu = 1 - target_percentile * 0.01
        # 获取完整的用户和物品嵌入
        # load u i embedding
        print('load u i embedding')
        ################################################################

        # user_emb = self.sess.run(self.user_embedding)
        # item_emb = self.sess.run(self.item_embedding)

        # 创建一个空列表，用于保存所有批次的数据（包括 R 和 O 标签）
        batch_size = 8192
        all_data_with_labels = []

        np.random.shuffle(self.train)
        total_samples = len(self.train)
        print('total_samples',total_samples)
        num_batches = (total_samples )// batch_size
        print('num_batches',num_batches)
        residuals = total_samples % batch_size
        print('residuals',residuals)
        if num_batches == 0:
            num_batches = 1
        for i in range(num_batches):   
            if i ==0:
                start=0
                end = batch_size
                end = end + residuals
            else:
                start = i * batch_size + residuals
                end = start + batch_size
            batch_data = self.train[start:end]
            print('batch_data',batch_data.shape)
            batch_pos_idx = np.where(batch_data[:, 2] == 1)[0]
            print('num_pos_idx',len(batch_pos_idx))
            batch_pos_data = batch_data[batch_pos_idx]
            pos_user_ids = batch_pos_data[:, 0].astype(np.int32)
            pos_item_ids = batch_pos_data[:, 1].astype(np.int32)
            pos_user_emb_selected = user_emb[pos_user_ids]
            pos_item_emb_selected = item_emb[pos_item_ids]
            
            # 提取当前批次中的用户和物品ID
            user_ids = batch_data[:, 0].astype(np.int32)
            item_ids = batch_data[:, 1].astype(np.int32)
            y_labels = batch_data[:, 2].astype(np.int32)  # 提取 Y 标签

            # 获取当前批次的用户和物品嵌入
            user_emb_selected = user_emb[user_ids]
            item_emb_selected = item_emb[item_ids]

            # 拼接用户和物品的嵌入
            pos_embeddings = np.concatenate([pos_user_emb_selected, pos_item_emb_selected], axis=1)
            embeddings = np.concatenate([user_emb_selected, item_emb_selected], axis=1)
            
            # 使用调整后的 nu 进行 OneClassSVM 分类
            print('fit ocsvm')
            ocsvm = OneClassSVM(kernel='rbf', nu=self.nu)
            ocsvm.fit(pos_embeddings)
            distances = ocsvm.decision_function(embeddings)
            print('assign labels')
            # 分配 R 和 O 标签，传入距离和 Y 标签
            R_O_labels = self.assign_labels(distances, y_labels)

            # 更新 batch_data，拼接生成的 R 和 O 标签
            batch_data_with_labels = np.hstack((batch_data, R_O_labels))

            # 将当前批次的数据添加到 all_data_with_labels 列表中
            all_data_with_labels.append(batch_data_with_labels)

        # 将所有批次的数据合并为一个完整的数据集，并返回
        all_data_with_labels = np.vstack(all_data_with_labels)
        # dump data
        np.save(f'/home/zhouchuan/unbiased-pairwise-rec-master/data/{self.data_name}_st_1/point/labeled_train_{self.percentile}_{self.HN_percentile}.npy', all_data_with_labels)
        #################################################################
        print('predict val')
        labeled_val = self.predict_classifier(self.nu, batch_size=8192)
        # dump data
        np.save(f'/home/zhouchuan/unbiased-pairwise-rec-master/data/{self.data_name}_st_1/point/labeled_val_{self.percentile}_{self.HN_percentile}.npy', labeled_val)
        print('finish')
        return all_data_with_labels, labeled_val

    def predict_classifier(self, adjusted_nu, batch_size=None, test_data=None):
        """
        Classify validation or test data using the previously calculated adjusted nu.
        - If `test_data` is None, the validation data (`self.val`) is used.
        - `adjusted_nu` must be provided from the training phase.
        """

        if test_data is None:
            data = self.val  # Use validation data if no test data is provided
        else:
            data = test_data  # Use test data if provided
            
            
        emb_path = f'../data/{self.data_name}_st_1/emb'
        user_emb = np.load(f'{emb_path}/user_embed.npy')
        item_emb = np.load(f'{emb_path}/item_embed.npy')
        # user_emb = self.sess.run(self.user_embedding)
        # item_emb = self.sess.run(self.item_embedding)

        # 创建一个空列表，用于保存所有批次的数据（包括 R 和 O 标签）
        batch_size = 8192
        all_data_with_labels = []

        np.random.shuffle(data)
        total_samples = len(data)
        num_batches = (total_samples) // batch_size
        residuals = total_samples % batch_size
        
        if num_batches == 0:
            num_batches = 1
        for i in range(num_batches):   
            if i ==0:
                start=0
                end = batch_size
                end = end + residuals
            else:
                start = i * batch_size + residuals
                end = start + batch_size
            batch_data = data[start:end]
            batch_pos_idx = np.where(batch_data[:, 2] == 1)[0]
            batch_pos_data = batch_data[batch_pos_idx]
            pos_user_ids = batch_pos_data[:, 0].astype(np.int32)
            pos_item_ids = batch_pos_data[:, 1].astype(np.int32)
            pos_user_emb_selected = user_emb[pos_user_ids]
            pos_item_emb_selected = item_emb[pos_item_ids]
            
            # 提取当前批次中的用户和物品ID
            user_ids = batch_data[:, 0].astype(np.int32)
            item_ids = batch_data[:, 1].astype(np.int32)
            y_labels = batch_data[:, 2].astype(np.int32)  # 提取 Y 标签

            # 获取当前批次的用户和物品嵌入
            user_emb_selected = user_emb[user_ids]
            item_emb_selected = item_emb[item_ids]

            # 拼接用户和物品的嵌入
            pos_embeddings = np.concatenate([pos_user_emb_selected, pos_item_emb_selected], axis=1)
            embeddings = np.concatenate([user_emb_selected, item_emb_selected], axis=1)
            
            # 使用调整后的 nu 进行 OneClassSVM 分类
            print('fit ocsvm')
            ocsvm = OneClassSVM(kernel='rbf', nu=self.nu)
            ocsvm.fit(pos_embeddings)
            distances = ocsvm.decision_function(embeddings)
            print('assign labels')
            # 分配 R 和 O 标签，传入距离和 Y 标签
            R_O_labels = self.assign_labels(distances, y_labels)

            # 更新 batch_data，拼接生成的 R 和 O 标签
            batch_data_with_labels = np.hstack((batch_data, R_O_labels))

            # 将当前批次的数据添加到 all_data_with_labels 列表中
            all_data_with_labels.append(batch_data_with_labels)

        # 将所有批次的数据合并为一个完整的数据集，并返回
        all_data_with_labels = np.vstack(all_data_with_labels)
        return all_data_with_labels

    def pdist2sq(self, A, B):
        """计算 A 和 B 之间的点对平方距离."""
        norm_A = tf.reduce_sum(tf.square(A), axis=1, keepdims=True)
        norm_B = tf.reduce_sum(tf.square(B), axis=1, keepdims=True)
        distances = norm_A - 2 * tf.matmul(A, B, transpose_b=True) + tf.transpose(norm_B)
        return distances
   

    def compute_loss(self):
        """计算总损失，包括交叉熵、BPR 和 IPM 损失"""

        # 使用 tf.equal 形式的条件
        DP_indices = tf.where(tf.logical_and(tf.logical_and(tf.equal(self.R_labels, 1), tf.equal(self.O_labels, 1)),
                                             tf.equal(self.Y_labels, 1)))[:, 0]
        HP_indices = tf.where(tf.logical_and(tf.logical_and(tf.equal(self.R_labels, 1), tf.equal(self.O_labels, 1)),
                                             tf.equal(self.Y_labels, 0)))[:, 0]
        HN_indices = tf.where(tf.logical_and(tf.equal(self.R_labels, 1), tf.equal(self.O_labels, 0)))[:, 0]
        UN_indices = tf.where(tf.logical_and(tf.equal(self.R_labels, 0), tf.equal(self.O_labels, -1)))[:, 0]

        # concate HP_indices, HN_indices, UN_indices, then randomly sample the same number of DP_indices
        concat_indices = tf.concat([HP_indices, HN_indices, UN_indices], axis=0)
        # tf.random.shuffle(concat_indices)

        # num_dp = tf.size(DP_indices)
        # neg_indices = tf.slice(concat_indices, [0], [num_dp])
        neg_pred = tf.gather(self.prediction, concat_indices)
        neg_mask = tf.equal(self.Y_labels, 0)
        
        

        DP_pred = tf.gather(self.prediction, DP_indices)
        HP_pred = tf.gather(self.prediction, HP_indices)
        HN_pred = tf.gather(self.prediction, HN_indices)
        UN_pred = tf.gather(self.prediction, UN_indices)

        # 交叉熵损失
        cross_entropy_loss_DP = self.binary_cross_entropy(tf.ones_like(DP_pred), DP_pred)
        cross_entropy_loss_HP = self.binary_cross_entropy(tf.zeros_like(HP_pred), HP_pred)

        # BPR损失：HP对UN，UN对HN
        # bpr_loss_1 = tf.cond(
        #     tf.logical_and(tf.greater(tf.size(HP_pred), 0), tf.greater(tf.size(UN_pred), 0)),
        #     lambda: self.bpr_loss(tf.reshape(UN_pred, [-1]), tf.reshape(HP_pred, [-1])),
        #     lambda: 0.0
        # )
        # bpr_loss_2 = tf.cond(
        #     tf.logical_and(tf.greater(tf.size(UN_pred), 0), tf.greater(tf.size(HN_pred), 0)),
        #     lambda: self.bpr_loss(tf.reshape(HN_pred, [-1]), tf.reshape(UN_pred, [-1])),
        #     lambda: 0.0
        # )

        # bpr_loss= self.bpr_loss(tf.reshape(DP_pred, [-1]), tf.reshape(neg_pred, [-1]))
        # bpr_loss = self.align_bpr_loss()
        

        # IPM损失
        # 取 R=1 和 R=0 的嵌入来计算 Wasserstein 距离
        R1_indices = tf.where(tf.equal(self.R_labels, 1))[:, 0]

        # R1 = tf.gather(self.R_labels, R1_indices)

        R1_embeddings = tf.gather(self.representation, R1_indices)
        # R0_embeddings = tf.gather(self.representation, R0_indices)
        O_labels = tf.gather(self.O_labels, R1_indices) 
        R1O1 = tf.where(tf.equal(O_labels, 1))[:, 0]
        # IPM Loss 1: R1 和 R0 之间的 Wasserstein 距离
        ###############################################################################################
        
        
        
        
        # ipm_loss_ro1, _ = self.wasserstein(X=self.representation, t=self.R_labels, p=0.5)
        # ipm_loss_r1_r0, _ = self.wasserstein(X=R1_embeddings, t=R1O1, p=0.5)
        
        
        
        
        
        
        # 判断四个损失是否为 NaN，如果是 NaN，则将其设置为 0
        ##########################################################################################
        cross_entropy_loss_DP = tf.where(tf.math.is_nan(cross_entropy_loss_DP), 0.0, cross_entropy_loss_DP)
        cross_entropy_loss_HP = tf.where(tf.math.is_nan(cross_entropy_loss_HP), 0.0, cross_entropy_loss_HP)
        # bpr_loss_1 = tf.where(tf.math.is_nan(bpr_loss_1), 0.0, bpr_loss_1)
        # bpr_loss_2 = tf.where(tf.math.is_nan(bpr_loss_2), 0.0, bpr_loss_2)
        
        reg_embeds = tf.nn.l2_loss(self.user_embedding)
        reg_embeds += tf.nn.l2_loss(self.item_embedding)
        
        
        # 收集正则化参数
        representation_vars = [
            var for var in tf.trainable_variables() 
            if 'representation_layer' in var.name and 'kernel' in var.name
        ]

        prediction_vars = [
            var for var in tf.trainable_variables() 
            if 'prediction_layer' in var.name and 'kernel' in var.name
        ]

        # 计算L2正则化项（平方和乘以系数）
        
        # with tf.name_scope("l2_regularization"):
        #     reg_mlp = tf.constant(0.0)
        #     for weights in self.representation_layer + self.prediction_layer:
        #         reg_mlp += tf.nn.l2_loss(weights)  # 计算权重的 L2 范数
        
        # l2_representation = tf.add_n([tf.nn.l2_loss(var) for var in representation_vars]) * 2  # 因为l2_loss返回sum/2
        # l2_prediction = tf.add_n([tf.nn.l2_loss(var) for var in prediction_vars]) * 2        
        # reg_mlp = l2_representation + l2_prediction
        
        
        
        # 总损失：确保所有损失都有效
        ##############################################################################################
        # self.unbiased_loss = #bpr_loss# + self.lambda2*(ipm_loss_ro1 + ipm_loss_r1_r0)
        self.unbiased_loss = cross_entropy_loss_DP + cross_entropy_loss_HP #+self.lambda1*(bpr_loss_1 + bpr_loss_2) + self.lambda2*(ipm_loss_ro1 + ipm_loss_r1_r0)
        
        
        self.loss = self.unbiased_loss# + self.wd1*reg_embeds + self.wd2*reg_mlp
        # print("total_loss", total_loss)

        # self.loss=total_loss

    def load_and_test_model(self, test_data=None, batch_size=128):
        """加载保存的最佳模型并在测试集上进行预测"""

        # 初始化保存器
        saver = tf.train.Saver()

        # 加载保存的最佳模型
        saver.restore(self.sess, f'../logs/{self.data_name}_st_1/Ours/best_model.ckpt')
        print("Best model restored.")

        if test_data is not None:
            predictions = []
            for start in range(0, len(test_data), batch_size):
                end = min(start + batch_size, len(test_data))
                batch_data = test_data[start:end]

                user_ids = batch_data[:, 0]
                item_ids = batch_data[:, 1]

                feed_dict = {
                    self.user_input: user_ids,
                    self.item_input: item_ids
                }

                # 获取预测结果
                preds = self.sess.run(self.prediction, feed_dict=feed_dict)
                predictions.append(preds)

            predictions = np.vstack(predictions)
            print("Predictions completed.")
            print('predictions.shape',predictions.shape)
            return predictions
        else:
            print("No test data provided.")
            return None

    def train_model(self, labeled_train, labeled_val, save_path="best_model.ckpt"):
        """使用 labeled_data 训练模型并监控验证集上的损失，存储验证集最优模型"""
        # 获取验证集的标签数据
        # self.sess.run(tf.global_variables_initializer())
        labeled_data=labeled_train

        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # # 初始化特定的变量


        # 创建一个 Saver 对象，用于保存模型
        saver = tf.train.Saver()

        best_val_loss = float('inf')  # 初始时，验证集的最小损失设为无穷大
        
        
        # Early stopping parameters
        last_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        
        # 训练循环
        for epoch in range(self.max_iters):
            labeled_data = np.array(labeled_data)
            # Shuffle the labeled_data using numpy
            np.random.shuffle(labeled_data)

            total_train_loss = 0.0
            total_val_loss = 0.0
            
            # Training on the training set
            pbar = tqdm(total=len(labeled_data), desc ="train:")
            for start in range(0, len(labeled_data), self.batch_size):
                pbar.update(self.batch_size)
                end = min(start + self.batch_size, len(labeled_data))
                batch_data = labeled_data[start:end]  # Direct slicing of labeled_data

                user_ids = batch_data[:, 0]
                item_ids = batch_data[:, 1]
                labels = batch_data[:, 2]
                R_labels = batch_data[:, 3]
                O_labels = batch_data[:, 4]

                feed_dict = {
                    self.user_input: user_ids,
                    self.item_input: item_ids,
                    self.Y_labels: labels,
                    self.R_labels: R_labels,
                    self.O_labels: O_labels
                }

                # loss = self.compute_loss()

                # l_rate = self.eta
                # optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)

                # minimize = optimizer.minimize(loss)

                # prediction, train_loss, _ = self.sess.run([self.prediction, loss, minimize], feed_dict=feed_dict)
                prediction, _, train_loss = self.sess.run(
                    [self.prediction, self.pred_optimizer, self.loss], 
                    feed_dict=feed_dict
                )
                # print("train_loss", train_loss)
                # pbar.write("train_loss: %f"%(train_loss, ))

                # minimize = self.optimizer.minimize(loss)
                # # 计算训练损失和执行优化
                # prediction, train_loss, _= self.sess.run([self.prediction, loss,  minimize], feed_dict=feed_dict)
                # self.optimizer.minimize(self.compute_loss())
                total_train_loss += train_loss
                
                val_loss = 0.0
                # 计算整个验证集上的损失
                for start in range(0, len(labeled_val), self.batch_size):
                    end = min(start + self.batch_size, len(labeled_val))
                    batch_data_val = np.array(labeled_val[start:end])  # Convert to NumPy array

                    user_ids_val = batch_data_val[:, 0]
                    item_ids_val = batch_data_val[:, 1]
                    labels_val = batch_data_val[:, 2]
                    R_labels_val = batch_data_val[:, 3]
                    O_labels_val = batch_data_val[:, 4]

                    feed_dict_val = {
                        self.user_input: user_ids_val,
                        self.item_input: item_ids_val,
                        self.Y_labels: labels_val,
                        self.R_labels: R_labels_val,
                        self.O_labels: O_labels_val
                    }

                    # 计算验证损失（不进行优化）
                    val_loss_0 = self.sess.run(self.unbiased_loss, feed_dict=feed_dict_val)
                    val_loss += val_loss_0
                total_val_loss += val_loss

            # 打印训练集和验证集的损失
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}: Training Loss = {total_train_loss:.4f}, Validation Loss = {total_val_loss:.4f}")

            # 如果验证损失比之前的最小值小，则保存当前模型
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                saver.save(self.sess, save_path)  # 保存当前模型的参数
                print(f"Best model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
            if total_val_loss < last_val_loss:
                last_val_loss = total_val_loss
            else:
                if epoch > 10:
                    patience_counter += 1
                last_val_loss = total_val_loss
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        return best_val_loss

