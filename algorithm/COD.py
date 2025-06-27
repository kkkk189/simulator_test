import random, os
from algorithm.alg_utility import *
from copy import deepcopy

class Estimator:
    """ build value network
    """
    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 env,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess
        self.n_valid_grid = env.n_valid_grids
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.M = env.M
        self.N = env.N
        self.scope = scope
        self.T = 144
        self.env = env

        # Writes Tensorboard summaries to disk
        self.summary_writer = None

        # Variable scope for "Estimator"
        with tf.variable_scope(scope):

            # Build the value function graph (NN for Critic)
            # with tf.variable_scope("value"):
            value_loss = self._build_value_model()

            # The Policy Graph (NN for Actor)
            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_mlp_policy()

            self.loss = actor_loss + .5 * value_loss - 10 * entropy

            # self.loss_gradients = tf.gradients(self.value_loss, tf.trainable_variables(scope=scope))
            # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            tf.summary.scalar("entropy", self.entropy),
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

        self.neighbors_list = []
        for idx, node_id in enumerate(env.target_grids):
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)

        # compute valid action mask.
        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.nodes
        self.valid_neighbor_grid_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.target_grids
        for grid_idx, grid_id in enumerate(env.target_grids):
            for neighbor_idx, neighbor in enumerate(self.env.nodes[grid_id].neighbors):
                if neighbor is None:
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0
                else:
                    node_index = neighbor.get_node_index()  # node_index in env.nodes
                    self.valid_neighbor_node_id[grid_idx, neighbor_idx] = node_index
                    self.valid_neighbor_grid_id[grid_idx, neighbor_idx] = env.target_grids.index(node_index)

            self.valid_neighbor_node_id[grid_idx, -1] = grid_id
            self.valid_neighbor_grid_id[grid_idx, -1] = grid_idx


    # Value Function Estimator (NN for Critic)
    def _build_value_model(self):

        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

        # Loss Learning Rate
        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 4 layers feed forward network.
        l1 = fc(X, "l1", 512, act=tf.nn.relu)
        l2 = fc(l1, "l2", 256, act=tf.nn.relu)
        l3 = fc(l2, "l3", 128, act=tf.nn.relu)
        l4 = fc(l3, "l3", 64, act=tf.nn.relu)
        # l1 = tf.layers.dense(X, 1024, tf.nn.sigmoid, trainable=trainable)
        # l2 = tf.layers.dense(l1, 512, tf.nn.sigmoid, trainable=trainable)
        # l3 = tf.layers.dense(l2, 32, tf.nn.sigmoid, trainable=trainable)
        self.value_output = fc(l4, "value_output", 1, act=tf.nn.relu)

        # self.losses = tf.square(self.y_pl - self.value_output)
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))

        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        return self.value_loss


    # Policy Estimator (NN for Actor)
    def _build_mlp_policy(self):

        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')
        self.neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")
        # this mask filter invalid actions and those action smaller than current grid value.

        # 3 layers feed forward network.
        l1 = fc(self.policy_state, "l1", 256, act=tf.nn.relu)
        l2 = fc(l1, "l2", 128, act=tf.nn.relu)
        l3 = fc(l2, "l3", 64, act=tf.nn.relu)

        # P logits
        self.logits = logits = fc(l3, "logits", self.action_dim,
                                  act=tf.nn.sigmoid) # + 1  # avoid valid_logits are all zeros
        # Q valid
        self.valid_logits = logits * self.neighbor_mask

        # Converting the logits to softmax probability (Action Probability - used to choose action)
        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        # Log of softmax is the policy gradient
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        # Negative of Score Function since we actually want to maximize it
        self.neglogprob = - self.logsoftmaxprob * self.ACTION
        # Gradient of policy
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        self.policy_loss = self.actor_loss - 0.01 * self.entropy

        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        return self.actor_loss, self.entropy


class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    def __init__(self,
                 target_id_states,
                 target_grids,
                 n_valid_grids):
        self.target_id_states = target_id_states  # valid grid index for driver and order distribution.
        self.target_grids = target_grids   # valid grid id [22, 24, ...]  504 grids
        self.n_valid_grids = n_valid_grids
        self.T = 144
        self.action_dim = 7
        self.extend_state = True

    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]
        return np.array(curr_s_new)

    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])
        max_order_num = np.max(curr_s[self.n_valid_grids:])
        if max_order_num == 0:
            max_order_num = 1
        if max_driver_num == 0:
            max_driver_num = 1
        curr_s_new = np.zeros_like(curr_s)
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num
        return curr_s_new

    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]
        return np.array(reward_node_new)

    def reward_wrapper(self, info, curr_s):
        """ reformat reward from env to the input of model.
        :param info: [node_reward(including neighbors), neighbor_reward]
        :param curr_s:  processed by utility_conver_states, same time step as info.
        :return:
        """

        info_reward = info[0]
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])
        devide = curr_s[:self.n_valid_grids]
        devide[devide == 0] = 1
        valid_nodes_reward = valid_nodes_reward/devide  # averaged rewards for drivers arriving this grid
        return valid_nodes_reward

    def compute_context(self, info):
        # compute context
        context = info.flatten()
        context = [context[idx] for idx in self.target_grids]
        return context

    def to_grid_states(self, curr_s, curr_city_time):
        """ extend global state to all agents' state.

        :param curr_s:
        :param curr_city_time: curr_s time step
        :return:
        """
        T = self.T

        # curr_s = self.utility_conver_states(curr_state)
        time_one_hot = np.zeros((T))
        time_one_hot[curr_city_time % T] = 1            # (1 x 144)
        onehot_grid_id = np.eye(self.n_valid_grids)     # Identity Matrix of size n_valid_grids

        s_grid = np.zeros((self.n_valid_grids, self.n_valid_grids * 3 + T))
        s_grid[:, :self.n_valid_grids * 2] = np.stack([curr_s] * self.n_valid_grids)
        s_grid[:, self.n_valid_grids * 2:self.n_valid_grids * 2 + T] = np.stack([time_one_hot] * self.n_valid_grids)
        s_grid[:, -self.n_valid_grids:] = onehot_grid_id

        return np.array(s_grid)

    def to_grid_rewards(self, node_reward):
        """
        :param node_reward: curr_city_time + 1 's reward
        :return:
        """
        return np.array(node_reward).reshape([-1, 1])

    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1
        return action_mat


class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        # self.next_states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []  # advantages

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, mask):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s),axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            # random.seed(0)
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    def sample(self):

        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        # random.seed(0)
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0


class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """

    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0  # current memory lens

    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            # random.seed(0)
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    def sample(self):

        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        # random.seed(0)
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0