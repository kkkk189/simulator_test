# -*- coding: utf-8 -*-
import pickle, sys
sys.path.append("../")

# from simulator.utilities import *
from algorithm.alg_utility import *
from simulator.envs import *
from shutil import copyfile

################## Load data ###################################
dir_prefix = "/mnt/research/linkaixi/AllData/dispatch/"
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print("log dir is {}".format(log_dir))

data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
order_time_dist = []
order_price_dist = []
mapped_matrix_int = pickle.load(open(data_dir+"mapped_matrix_int.pkl", 'rb'))
order_num_dist = pickle.load(open(data_dir+"order_num_dist", 'rb'))
idle_driver_dist_time = pickle.load(open(data_dir+"idle_driver_dist_time", 'rb'))
idle_driver_location_mat = pickle.load(open(data_dir+"idle_driver_location_mat", 'rb'))
target_ids = pickle.load(open(data_dir+"target_grid_id.pkl", 'rb'))
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat", 'rb'))
order_filename = dir_prefix + "dispatch_realdata/orders/all_orders_target"
order_real = pickle.load(open(order_filename, 'rb'))
M, N = mapped_matrix_int.shape
print("finish load data")


################## Initialize env ###################################
n_side = 6
GAMMA = 0.9
l_max = 9

env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location_mat,
               order_time_dist, order_price_dist,
               l_max, M, N, n_side, 1/28.0, order_real, onoff_driver_location_mat)

log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)

# Adds double space for driver and order for each grid
temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()


curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
# Filters the invalid grids
curr_s = utility_conver_states(curr_s, target_id_states)
print("******************* Finish generating one day order **********************")


print("******************* Starting training Deep actor critic **********************")
from algorithm.COD import *



MAX_ITER = 50
is_plot_figure = False
city_time_start = 0
EP_LEN = 144

# Model Input Parameters
city_time_end = city_time_start + EP_LEN
epsilon = 0.5
gamma = 0.9
learning_rate = 1e-3

prev_episode_reward = 0

# Parameters to track the output (Rewards) of our Models
all_rewards = []
order_response_rate_episode = []
value_table_sum = []
episode_rewards = []
episode_conflicts_drivers = []
record_all_order_response_rate = []

# Time step, Action Dimension & State Dimension
T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T


# Starting a session and building a new Model Object
# tf.reset_default_graph()
sess = tf.Session()
tf.set_random_seed(1)
q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=log_dir)


sess.run(tf.global_variables_initializer())

# Creating the Replay Memories and State Processor Object
replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)


# Creating the TF saver object to save our Model Parameters
restore = True
saver = tf.train.Saver()


# record_curr_state = []
# record_actions = []
save_random_seed = []
episode_dispatched_drivers = []
global_step1 = 0
global_step2 = 0
RATIO = 1

for n_iter in np.arange(25):
    RANDOM_SEED = n_iter + MAX_ITER - 10
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    episode_reward = 0
    num_dispatched_drivers = 0

    # Pre-dispatch Procedure
    # reset env
    is_regenerate_order = 1
    # Gets the current observations: Idle drivers and orders for M * N grids (2 x M x N)
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    # Assigns orders to drivers (2 step process), returns remaining drivers and orders (2 x M x N)
    info = env.step_pre_order_assigin(curr_state)
    # Returns only the remaining drivers in valid grids
    context = stateprocessor.compute_context(info)
    # Returns the drivers and orders in valid grids from the current observation/state (1 x drivers + orders)
    curr_s = stateprocessor.utility_conver_states(curr_state)
    # Normalized Curr_s (1 x drivers x orders)
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # All Agent state (Global state + Grid ID = n_valid x n_valid * 3 + T)
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # record rewards to update the value table
    episodes_immediate_rewards = []
    num_conflicts_drivers = []
    curr_num_actions = []
    order_response_rates = []
    for ii in np.arange(EP_LEN + 1):
        # record_curr_state.append(curr_state)

        # INPUT: state,  OUTPUT: action
        # The action is basically the driver re-allocation
        # action_tuple is the re-allocation of drivers (n_valid * (action_dim - 1) x 3)
        # (start_node_id, end_node_id, num_driver)
        action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
        curr_state_value, curr_neighbor_mask, next_state_ids = q_estimator.action(s_grid, context, epsilon)
        # a0

        # ONE STEP: r0
        next_state, r, info = env.step(action_tuple, 2)