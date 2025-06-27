import numpy as np
from abc import ABCMeta, abstractmethod
from simulator.utilities import *

# 定义了一个从中抽取样本（比如订单）的分布的通用接口。它作为其他具体分布类（如正态分布、均匀分布等）的基础，要求子类实现如何生成样本的具体方法。
class Distribution():
    ''' Define the distribution from which sample the orders'''
    __metaclass__ = ABCMeta  # python 2.7
    @abstractmethod
    def sample(self):
        pass

# 继承自抽象基类 Distribution 的一个具体实现，表示泊松分布。它通过实现 sample 方法，允许从泊松分布中抽取样本。
class PoissonDistribution(Distribution):

    def __init__(self, lam):
        self._lambda = lam

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.poisson(self._lambda, 1)[0]

# 是继承自抽象基类 Distribution 的一个具体实现，表示正态分布（或称高斯分布）。它通过实现 sample 方法，允许从正态分布中抽取样本。
class GaussianDistribution(Distribution):

    def __init__(self, args):
        mu, sigma = args
        self.mu = mu        # mean
        self.sigma = sigma  # standard deviation

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.normal(self.mu, self.sigma, 1)[0]

# 这个类 Node 表示网格中的一个节点，具有多个属性和方法，主要用于管理该节点的订单和驾驶员信息。它涉及到订单的生成、分配、驾驶员的管理等操作，常用于模拟和调度任务。
class Node(object):
    __slots__ = ('neighbors', '_index', 'orders', 'drivers',
                 'order_num', 'idle_driver_num', 'offline_driver_num',
                 'order_generator', 
                 'n_side', 'layers_neighbors', 'layers_neighbors_id')
    '''
    这行代码定义了 Node 类实例的属性，限制了实例只能拥有这些属性：
    neighbors：节点的邻居节点列表。
    _index：该节点的唯一索引。
    orders：该节点的订单列表。
    drivers：一个字典，存储节点上的驾驶员。
    order_num：当前节点的订单数量。
    idle_driver_num：当前节点上闲置的驾驶员数量。
    offline_driver_num：当前节点上离线的驾驶员数量。
    order_generator：用于生成订单的分布生成器。
    n_side：表示节点拓扑结构的维度（例如，六边形网格的每个节点与多少个邻居连接）。
    layers_neighbors：当前节点的各层邻居的坐标。
    layers_neighbors_id：当前节点的各层邻居的节点索引。
    '''

    #初始化一个 Node 类的实例。该构造函数定义了该节点的初始状态，包括节点的唯一索引、邻居、订单、驾驶员等信息
    def __init__(self, index):
        # private
        self._index = index   # unique node index. 节点的唯一标识符。每个节点都有一个不同的 index，用于在整个系统中唯一标识该节点。

        # public
        self.neighbors = []  # a list of nodes that neighboring the Nodes 是一个空列表，表示该节点的邻居节点。每个节点可能有多个邻居，这里用一个列表来存储它们。
        self.orders = []     # a list of orders 是一个空列表，表示该节点的订单列表。每个节点可以接收并生成多个订单。
        self.drivers = {}    # a dictionary of driver objects contained in this node 是一个空字典，用于存储该节点上的驾驶员对象。字典的键通常是驾驶员的 ID，值是与该驾驶员相关的对象或信息。
        self.order_num = 0   # 是该节点的订单数量。初始化为 0，表示节点没有任何订单。
        self.idle_driver_num = 0  # number of idle drivers in this node 表示该节点上闲置的驾驶员数量。初始化为 0，表示当前没有闲置驾驶员。
        self.offline_driver_num = 0 # 表示该节点上离线驾驶员的数量。初始化为 0，表示当前没有离线驾驶员。
        self.order_generator = None  # 是一个用于生成订单的分布生成器。初始化为 None，表示当前没有定义订单生成的机制。后续可能会设置为某种分布生成器，如泊松分布或高斯分布。

        self.n_side = 0      # the topology is a n-sided map 表示节点的拓扑结构是一个多少边的图。初始化为 0，表示该节点的拓扑结构未定义或未知。
        self.layers_neighbors = []  # layer 1 indices: layers_neighbors[0] = [[1,1], [0, 1], ...],  是一个空列表，用来存储该节点的各层邻居节点的坐标。
        # layer 2 indices layers_neighbors[1]     层次化的邻居关系通常用于多层网格或区域划分模型中。每一层的邻居会存储在不同的列表中，layers_neighbors[0] 存储第 1 层邻居，layers_neighbors[1] 存储第 2 层邻居，以此类推。
        self.layers_neighbors_id = [] # layer 1: layers_neighbors_id[0] = [2, 1,.]  是一个空列表，用来存储该节点的各层邻居的 ID。存储的是 layers_neighbors 中各层邻居的节点索引。这些 ID 会用来定位节点的具体位置，便于访问和处理邻居节点。


    # 清空当前节点的订单列表 orders。每个节点可能接收和处理多个订单，调用 clean_node 后，这些订单都会被移除
    def clean_node(self):
        self.orders = []   # 清空当前节点的订单列表 orders。每个节点可能接收和处理多个订单，调用 clean_node 后，这些订单都会被移除。
        self.order_num = 0 # 重置当前节点的订单数量 order_num 为 0。由于所有订单被清除，节点的订单数量也应归零。
        self.drivers = {}  # 清空当前节点的驾驶员字典 drivers。该字典存储了该节点上所有驾驶员的信息，调用 clean_node 后，字典将被清空。
        self.idle_driver_num = 0  # 重置闲置驾驶员数量 idle_driver_num 为 0。由于所有驾驶员信息被清除，闲置驾驶员数量也应归零。
        self.offline_driver_num = 0 # 重置离线驾驶员数量 offline_driver_num 为 0。和 idle_driver_num 类似，离线驾驶员的数量也被重置。


    # 计算和获取当前节点的各层邻居节点的信息，并将这些邻居节点的 ID 存储在 layers_neighbors_id 属性中。这些邻居节点根据层次结构进行分类，每一层存储不同距离的邻居节点。
    def get_layers_neighbors(self, l_max, M, N, env):

        x, y = ids_1dto2d(self.get_node_index(), M, N) # 获取当前节点的二维坐标 (x, y)，从一维索引转换为二维坐标。  self.get_node_index() 获取当前节点的唯一一维索引。ids_1dto2d 将该一维索引转换为二维坐标 (x, y)，M 和 N 分别是网格的行数和列数。
        self.layers_neighbors = get_layers_neighbors(x, y, l_max, M, N)  # 获取当前节点周围的各层邻居节点的坐标

        # # 遍历每一层的邻居
        for layer_neighbors in self.layers_neighbors:
            temp = [] # 用于存储有效的邻居节点 ID
            for item in layer_neighbors:
                x, y = item   # 获取每个邻居节点的坐标
                node_id = ids_2dto1d(x, y, M, N)  # 将邻居的二维坐标转换为一维节点索引

                # 检查该节点在环境中是否存在
                if env.nodes[node_id] is not None:
                    temp.append(node_id) # 如果该节点存在，将其 ID 添加到临时列表中

            self.layers_neighbors_id.append(temp) # 将每一层的有效邻居 ID 列表添加到 layers_neighbors_id 中


    # 返回当前节点的唯一标识符（索引）。它的功能就是访问并返回节点实例的私有属性 _index
    def get_node_index(self):
        return self._index

    # 返回当前节点上闲置驾驶员的数量。具体来说，它返回 self.idle_driver_num，即存储在节点中的闲置驾驶员的数量。（有在线和不在线）
    def get_driver_numbers(self):
        return self.idle_driver_num

    # 遍历当前节点的所有驾驶员，统计并返回该节点上闲置且在线的驾驶员数量。它的实现通过遍历 self.drivers 字典，检查每个驾驶员的状态（是否闲置和在线），然后统计符合条件的驾驶员数量
    def get_idle_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is True:
                temp_idle_driver += 1
        return temp_idle_driver

    # 遍历当前节点的所有驾驶员，统计并返回当前节点上 离线且闲置的驾驶员 的数量。它通过遍历 self.drivers 字典，检查每个驾驶员的状态（是否离线且闲置），然后统计符合条件的驾驶员数量。
    def get_off_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is False:
                temp_idle_driver += 1
        return temp_idle_driver


    # 用于为当前节点选择并设置一个订单生成器（order_generator）。根据传入的分布类型（distribution），它会创建相应的分布实例（如 泊松分布 或 高斯分布），并将其赋值给 self.order_generator。这个生成器用于在后续生成订单时产生随机样本。
    def order_distribution(self, distribution, dis_paras):

        if distribution == 'Poisson':
            self.order_generator = PoissonDistribution(dis_paras)
        elif distribution == 'Gaussian':
            self.order_generator = GaussianDistribution(dis_paras)
        else:
            pass


    # 用于在每个时间步骤生成新订单。它通过调用预设的订单生成器（如泊松分布或高斯分布）来确定在当前时间步骤生成的订单数量，并根据随机生成的价格、目的地等信息为每个订单生成相关数据。最终，生成的订单被添加到节点的订单列表中。
    def generate_order_random(self, city_time, nodes, seed):
        """Generate new orders at each time step
        """

        # 使用订单生成器生成订单数量（例如从泊松分布或高斯分布中抽取样本）
        num_order_t = self.order_generator.sample(seed)
        self.order_num += num_order_t

        # 生成每一个订单
        for ii in np.arange(num_order_t):
            price = np.random.normal(50, 5, 1)[0]  # 生成一个订单的价格，价格服从正态分布，均值50，标准差5
            price = 10 if price < 0 else price     # 如果价格小于0，将价格设为10（防止价格为负）

            current_node_id = self.get_node_index()  # 获取当前节点的ID
            destination_node = [kk for kk in np.arange(len(nodes)) if kk != current_node_id]   # 生成一个目的地节点列表，排除当前节点本身。 np.arange(len(nodes)) 生成一个从 0 到 len(nodes)-1 的整数数组。 len(nodes) 获取节点列表 nodes 的长度，即节点的总数。

            self.orders.append(Order(nodes[current_node_id],                                   # 当前节点（订单起始位置）
                                     nodes[np.random.choice(destination_node, 1)[0]],          # 目标节点（订单目的地）
                                     city_time,                                                # 当前时间（订单生成时间）
                                     # city_time + np.random.choice(5, 1)[0]+1,
                                     np.random.choice(2, 1)[0]+1,  # duration                  # 订单持续时间
                                     price, 1))                                                # 订单价格，1：标记为未完成的订单
        return



    # 在每个时间步骤生成新的订单。与 generate_order_random 方法不同的是，它通过考虑订单的持续时间分布和价格分布来生成更为真实的订单数据。订单的持续时间、价格、目的地等信息都受到不同的概率分布的影响，并且使用了当前节点和邻居节点的信息。
    def generate_order_real(self, l_max, order_time_dist, order_price_dist, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        num_order_t = self.order_generator.sample(seed)
        self.order_num += num_order_t

        for ii in np.arange(num_order_t):

            if l_max == 1:
                duration = 1
            else:

                duration = np.random.choice(np.arange(1, l_max+1), p=order_time_dist)
            price_mean, price_std = order_price_dist[duration-1]
            price = np.random.normal(price_mean, price_std, 1)[0]
            price = price if price > 0 else price_mean

            current_node_id = self.get_node_index()
            destination_node = []
            for jj in np.arange(duration):
                for kk in self.layers_neighbors_id[jj]:
                    if nodes[kk] is not None:
                        destination_node.append(kk)
            self.orders.append(Order(nodes[current_node_id],
                                     nodes[np.random.choice(destination_node, 1)[0]],
                                     city_time,
                                     duration,
                                     price, 1))
        return



    # 向当前节点的订单列表中添加一个新的订单。与其他订单生成方法不同，add_order_real 允许手动指定订单的目的地、持续时间和价格等参数，而不依赖于随机生成的分布。这个方法适用于需要明确指定订单详细信息的场景。
    def add_order_real(self, city_time, destination_node, duration, price):
        current_node_id = self.get_node_index()  # 获取当前节点的唯一标识符 current_node_id，通过调用 self.get_node_index() 方法获得。

        self.orders.append(Order(self,                   # self，即当前节点对象本身。它代表订单的起点。
                                 destination_node,       # destination_node，这是订单的目标节点。
                                 city_time,              # city_time，表示订单生成时的时间戳。
                                 duration,               # duration，订单的持续时间（如服务时间）。
                                 price, 0))              # price，订单的价格（例如运输费用）。0 表示订单是未完成的（0 表示未完成，1 表示完成）。这里默认订单是未完成的。
        self.order_num += 1



    # 设置当前节点的邻居节点列表，并更新当前节点的拓扑结构信息。该方法接受一个 nodes_list 参数，它表示当前节点的邻居节点列表，并根据这个列表来更新节点的邻居信息以及拓扑结构。
    def set_neighbors(self, nodes_list):
        self.neighbors = nodes_list    # nodes_list 是传入的参数，表示当前节点的邻居节点列表。通常，邻居节点是当前节点周围直接连接的其他节点。
        self.n_side = len(nodes_list)



    # 随机移除一个闲置且在线的驾驶员。该方法遍历当前节点的所有驾驶员，找到第一个满足 "闲置且在线" 状态的驾驶员，并将其移除。移除的驾驶员的 ID 会被返回。
    def remove_idle_driver_random(self):
        """Randomly remove one idle driver from current grid"""
        removed_driver_id = "NA"

        for key, item in self.drivers.items():      # 遍历当前节点的所有驾驶员
            if item.onservice is False and item.online is True:
                self.remove_driver(key)           # 移除该驾驶员
                removed_driver_id = key           # 记录移除的驾驶员 ID
            if removed_driver_id != "NA":         # 如果已经移除一个驾驶员，跳出循环
                break
        assert removed_driver_id != "NA"          # 确保至少移除了一名闲置且在线的驾驶员
        return removed_driver_id                  # 返回移除的驾驶员 ID



    # 随机将一个闲置且在线的驾驶员设置为离线。该方法遍历当前节点的所有驾驶员，找到第一个闲置且在线的驾驶员，并将其状态更改为离线（online=False）。该方法返回被设置为离线的驾驶员的 ID。
    def set_idle_driver_offline_random(self):
        """Randomly set one idle driver offline"""
        removed_driver_id = "NA"    # 初始化变量，用于记录被设置为离线的驾驶员 ID
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is True:    # 判断驾驶员是否闲置且在线
                item.set_offline()                                # 如果是，调用 `set_offline` 方法将其设置为离线
                removed_driver_id = key                            # 记录该驾驶员的 ID
            if removed_driver_id != "NA":                          # 如果已经成功设置一个驾驶员为离线，跳出循环
                break
        assert removed_driver_id != "NA"                           # 确保至少有一个闲置且在线的驾驶员被设置为离线
        return removed_driver_id                                   # 返回被设置为离线的驾驶员 ID



    # 将 离线且闲置的驾驶员 设置为 在线 状态，并返回该驾驶员的 ID。这个方法遍历当前节点的所有驾驶员，找到第一个符合 离线且闲置 条件的驾驶员，并将其状态更改为在线（online=True）。
    def set_offline_driver_online(self):

        online_driver_id = "NA"    # 初始化变量，用于记录将被设置为在线的驾驶员 ID
        for key, item in self.drivers.items():                 # 遍历当前节点的所有驾驶员
            if item.onservice is False and item.online is False:    # 判断驾驶员是否闲置且离线
                item.set_online()                                 # 如果是，调用 `set_online` 方法将其设置为在线
                online_driver_id = key                            # 记录该驾驶员的 ID
            if online_driver_id != "NA":                          # 如果已经成功设置一个驾驶员为在线，跳出循环
                break
        assert online_driver_id != "NA"                           # 确保至少有一个离线且闲置的驾驶员被设置为在线
        return online_driver_id                                   # 返回被设置为在线的驾驶员 ID



    # 随机获取一个驾驶员，但在这段代码中，实际上只随机选择了一个驾驶员（通常是字典中的第一个）。这个方法通过断言确保至少有一个闲置的驾驶员，并返回一个随机的驾驶员对象。
    def get_driver_random(self):
        """Randomly get one driver"""
        assert self.idle_driver_num > 0   # 确保有至少一个闲置驾驶员
        get_driver_id = 0                 # 初始化变量，暂时用 0 存储选中的驾驶员 ID
        for key in self.drivers.keys():    # 遍历当前节点的所有驾驶员 ID
            get_driver_id = key               # 将第一个找到的驾驶员 ID 赋值给 `get_driver_id`
            break                            # 跳出循环（只选择第一个驾驶员）
        return self.drivers[get_driver_id]       # 返回所选驾驶员的对象，self.drivers 字典存储了所有驾驶员，self.drivers[get_driver_id] 返回与该 ID 对应的驾驶员对象。



    # 从当前节点中移除一个驾驶员。它通过使用 pop 方法从 drivers 字典中删除指定的驾驶员，并相应地更新节点的闲置驾驶员数量（idle_driver_num）。如果尝试移除的驾驶员不存在，它会引发一个异常。
    def remove_driver(self, driver_id):

        removed_driver = self.drivers.pop(driver_id, None)  # 从字典中移除指定驾驶员
        self.idle_driver_num -= 1                           # 更新闲置驾驶员数量
        if removed_driver is None:
            raise ValueError('Nodes.remove_driver: Remove a driver that is not in this node')

        return removed_driver    # 返回被移除的驾驶员对象


    # 向当前节点添加一个驾驶员。它将一个新的驾驶员对象添加到节点的 drivers 字典中，并更新闲置驾驶员的数量 idle_driver_num。
    def add_driver(self, driver_id, driver):
        self.drivers[driver_id] = driver    # 将驾驶员添加到字典中
        self.idle_driver_num += 1    # 更新闲置驾驶员的数量



    # 从当前节点的订单列表中移除 超时未完成及已完成 的订单。它根据当前时间 city_time 判断每个订单的状态，移除那些已经超时未被完成及已完成的订单，并更新订单列表和订单数量。
    def remove_unfinished_order(self, city_time):
        un_finished_order_index = []  # 初始化一个列表，用来存储未完成订单的索引
        for idx, o in enumerate(self.orders):     # 遍历当前节点的所有订单
            # order un served 判断订单是否未被服务
            if o.get_wait_time()+o.get_begin_time() < city_time:    # 如果订单等待时间 + 开始时间小于当前时间，表示订单未完成
                un_finished_order_index.append(idx)

            # order completed  判断订单是否已经完成   o.get_assigned_time() 获取订单的分配时间（订单被分配给驾驶员的时间）。 o.get_duration() 获取订单的持续时间（订单的服务时间）。
            if o.get_assigned_time() + o.get_duration() == city_time and o.get_assigned_time() != -1:  # 如果订单的分配时间加上持续时间等于当前时间，并且订单的分配时间不为 -1（意味着订单已经被分配给了驾驶员），则表示订单已经完成，将该订单的索引 idx 加入 un_finished_order_index 列表中。
                un_finished_order_index.append(idx)

        if len(un_finished_order_index) != 0:
            # remove unfinished orders
            self.orders = [i for j, i in enumerate(self.orders) if j not in un_finished_order_index]
            self.order_num = len(self.orders)



    # 一个简单的订单分配方法，用于将订单分配给闲置且在线的驾驶员。该方法尝试为每个订单分配一个驾驶员，更新订单的状态，并计算分配的奖励。最后，它会返回已分配订单的奖励、所有订单的数量和已分配订单的数量。
    def simple_order_assign(self, city_time, city):
        reward = 0   # 初始化奖励为 0
        num_assigned_order = min(self.order_num, self.idle_driver_num)   # 计算需要分配的订单数量
        served_order_index = []                     # 用于记录已分配的订单的索引
        for idx in np.arange(num_assigned_order):      # 遍历所有待分配的订单
            order_to_serve = self.orders[idx]          # 获取当前需要分配的订单对象
            order_to_serve.set_assigned_time(city_time) # 设置订单的分配时间
            self.order_num -= 1                        # 更新订单数量，减少一个待分配订单
            reward += order_to_serve.get_price()       # 累加订单的价格到奖励
            served_order_index.append(idx)             # 将订单的索引添加到已分配订单列表中

            for key, assigned_driver in self.drivers.items():     # 遍历所有驾驶员，找一个闲置且在线的驾驶员来分配订单
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    assigned_driver.take_order(order_to_serve)          # 将订单分配给该驾驶员
                    removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)  # 移除分配的驾驶员
                    assert removed_driver is not None  # 确保成功移除驾驶员
                    city.n_drivers -= 1      # 城市中可用驾驶员数减少
                    break    # 找到一个闲置且在线的驾驶员后，跳出循环

        all_order_num = len(self.orders)   # 获取当前节点的所有订单数量
        finished_order_num = len(served_order_index)  # 获取已分配的订单数量

        # remove served orders 从订单列表中移除已分配的订单
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)   # 确保当前节点的订单数量更新正确

        return reward, all_order_num, finished_order_num  # 返回奖励、所有订单数量和已分配订单数量


    # 一个 真实的订单分配方法，用于将订单分配给闲置且在线的驾驶员，同时考虑订单的目标区域（目的地）和其他动态因素。与之前的方法不同，simple_order_assign_real 方法会根据订单的 目的地 来决定是否将其分配给驾驶员，如果订单的目的地不在目标区域内，驾驶员会被设置为离线。
    def simple_order_assign_real(self, city_time, city):

        reward = 0
        num_assigned_order = min(self.order_num, self.idle_driver_num)
        served_order_index = []
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)
            for key, assigned_driver in self.drivers.items():
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    if order_to_serve.get_end_position() is not None:
                        assigned_driver.take_order(order_to_serve)
                        removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
                        assert removed_driver is not None
                    else:
                        assigned_driver.set_offline()  # order destination is not in target region
                    city.n_drivers -= 1
                    break

        all_order_num = len(self.orders)
        finished_order_num = len(served_order_index)

        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)

        return reward, all_order_num, finished_order_num



    # 在当前节点与其邻居节点之间广播并更新订单分配。该方法的目的是将当前节点的订单分配给邻居节点的驾驶员，并更新邻居节点的奖励信息。它还会更新每个邻居节点的订单完成情况和当前节点的订单状态。
    def simple_order_assign_broadcast_update(self, city, neighbor_node_reward):

        assert self.idle_driver_num == 0   # 确保当前节点没有闲置的驾驶员
        reward = 0
        num_finished_orders = 0    # 初始化已完成订单的数量为 0
        for neighbor_node in self.neighbors:   # 遍历所有邻居节点
            if neighbor_node is not None and neighbor_node.idle_driver_num > 0:     # 判断邻居节点是否有效且有闲置驾驶员
                num_assigned_order = min(self.order_num, neighbor_node.idle_driver_num)     # 计算可以分配给邻居节点的订单数量
                rr = self.utility_assign_orders_neighbor(city, neighbor_node, num_assigned_order)   # 调用 utility_assign_orders_neighbor 方法为邻居节点分配订单
                reward += rr    # 累加奖励
                neighbor_node_reward[neighbor_node.get_node_index()] += rr   # 更新邻居节点的奖励信息
                num_finished_orders += num_assigned_order  # 更新已完成的订单数量
            if self.order_num == 0:    # 如果当前节点的订单已经分配完毕，跳出循环
                break

        assert self.order_num == len(self.orders)    # 确保当前节点的订单数量更新正确
        return reward, num_finished_orders    # 返回奖励和已完成的订单数量



    # 将订单分配给 邻居节点 的驾驶员，并计算分配的奖励。它通过遍历订单列表，选择合适的驾驶员来接收订单，更新订单状态，并计算奖励。方法完成后，还会更新当前节点和邻居节点的状态，移除已分配的订单，并返回该次分配的奖励。
    def utility_assign_orders_neighbor(self, city, neighbor_node, num_assigned_order):

        served_order_index = []  # 用于记录已分配的订单索引
        reward = 0
        curr_city_time = city.city_time   # 获取当前城市时间
        for idx in np.arange(num_assigned_order):       # 遍历需要分配的订单数量
            order_to_serve = self.orders[idx]           # 获取当前订单对象
            order_to_serve.set_assigned_time(curr_city_time)    # 设置订单的分配时间为当前城市时间
            self.order_num -= 1                         # 更新当前节点的订单数量
            reward += order_to_serve.get_price()        # 累加订单的价格到奖励中
            served_order_index.append(idx)              # 将已分配的订单索引添加到列表中


            for key, assigned_driver in neighbor_node.drivers.items():    # 遍历邻居节点的所有驾驶员
                if assigned_driver.onservice is False and assigned_driver.online is True:   # 判断驾驶员是否闲置且在线
                    if order_to_serve.get_end_position() is not None:      # 检查订单是否有明确的目的地
                        assigned_driver.take_order(order_to_serve)        # 将订单分配给该驾驶员
                        removed_driver = neighbor_node.drivers.pop(assigned_driver.get_driver_id(), None)      # 移除该驾驶员
                        assert removed_driver is not None      # 确保成功移除驾驶员
                    else:
                        assigned_driver.set_offline()       # 如果订单没有目标位置，将该驾驶员设置为离线
                    city.n_drivers -= 1                     # 更新城市中的可用驾驶员数量
                    break                                    # 一旦找到合适的驾驶员，跳出内层循环

        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]    # 移除已分配的订单
        assert self.order_num == len(self.orders)     # 确保当前节点的订单数量更新正确

        return reward    # 返回本次订单分配的奖励

# 表示一个驾驶员对象，管理其状态、位置、服务的订单等信息。该类中的方法控制驾驶员的状态（例如在线、离线、是否在服务中），更新城市时间，处理订单等。
class Driver(object):
    __slots__ = ("online", "onservice", 'order', 'node', 'city_time', '_driver_id')

    def __init__(self, driver_id):
        self.online = True
        self.onservice = False
        self.order = None     # the order this driver is serving
        self.node = None      # the node that contain this driver.
        self.city_time = 0  # track the current system time

        # private
        self._driver_id = driver_id  # unique driver id.

        '''
        初始化驾驶员对象的属性：
        online：表示驾驶员是否在线，初始化为 True。
        onservice：表示驾驶员是否正在服务，初始化为 False（即驾驶员开始时没有在服务中）。
        order：存储该驾驶员正在服务的订单，初始化为 None。
        node：表示驾驶员所在的节点，初始化为 None。
        city_time：记录当前城市时间，初始化为 0。
        _driver_id：每个驾驶员都有一个唯一的 ID。
        '''
    # 设置驾驶员的当前位置为指定的节点
    def set_position(self, node):
        self.node = node

    # 设置驾驶员开始服务的订单。该方法将传入的 order 对象赋值给驾驶员的 order 属性，表示该驾驶员正在处理这个订单。
    def set_order_start(self, order):
        self.order = order

    # 标记驾驶员完成当前服务的订单。该方法会清空驾驶员的订单信息，并将驾驶员的 onservice 状态设置为 False，表示该驾驶员已经完成订单服务。
    def set_order_finish(self):
        self.order = None
        self.onservice = False

    # 用于返回 驾驶员的唯一 ID。
    def get_driver_id(self):
        return self._driver_id

    # 用于 更新城市时间。它将当前的 city_time 增加 1，模拟系统中时间的流逝。
    def update_city_time(self):
        self.city_time += 1

    # 用于 设置城市时间。它直接将传入的 city_time 值赋给驾驶员的 city_time 属性。
    def set_city_time(self, city_time):
        self.city_time = city_time


    # 用于将 驾驶员设置为离线状态。该方法首先确保驾驶员处于闲置且在线的状态（onservice 为 False 且 online 为 True），然后将其状态设置为 离线，并更新相关节点中驾驶员的数量。
    def set_offline(self):
        assert self.onservice is False and self.online is True
        self.online = False
        self.node.idle_driver_num -= 1
        self.node.offline_driver_num += 1



    # 用于将驾驶员设置为 离线，并且确保驾驶员没有在服务中。该方法通常在调度系统中使用，当系统需要将一个驾驶员标记为离线以进行任务调度时，调用此方法。
    def set_offline_for_start_dispatch(self):

        assert self.onservice is False
        self.online = False


    # 将驾驶员设置为在线状态，并更新当前节点中驾驶员的数量。该方法确保驾驶员没有在服务中，并通过更新节点的 idle_driver_num（闲置驾驶员数量）和 offline_driver_num（离线驾驶员数量）来反映驾驶员状态的变化。
    def set_online(self):
        assert self.onservice is False
        self.online = True
        self.node.idle_driver_num += 1
        self.node.offline_driver_num -= 1


    # 将驾驶员设置为 在线状态，并确保该驾驶员 没有在服务中。这个方法通常在订单完成后的阶段使用，意味着驾驶员完成了当前订单的服务，可以准备接收新的订单。
    def set_online_for_finish_dispatch(self):

        self.online = True
        assert self.onservice is False


    # 用于 接收订单。当一个驾驶员接到订单时，该方法会更新驾驶员的状态，标记订单为已接收并开始服务，同时更新节点中的闲置驾驶员数量。
    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
        """
        assert self.online == True
        self.set_order_start(order)
        self.onservice = True
        self.node.idle_driver_num -= 1


    # 控制和更新驾驶员的状态，尤其是与订单的完成时间和城市时间相关的状态。这个方法每次被调用时，都会检查驾驶员当前的状态，并根据城市时间来更新驾驶员的状态。
    def status_control_eachtime(self, city):

        assert self.city_time == city.city_time  # 确保驾驶员的城市时间与城市对象的时间一致
        if self.onservice is True:      # 如果驾驶员正在服务中
            assert self.online is True    # 确保驾驶员是在线的
            order_end_time = self.order.get_assigned_time() + self.order.get_duration()   # 计算订单的结束时间
            if self.city_time == order_end_time:       # 如果当前时间等于订单的结束时间
                self.set_position(self.order.get_end_position())       # 设置驾驶员的位置为订单的结束位置
                self.set_order_finish()                                # 完成订单，结束服务
                self.node.add_driver(self._driver_id, self)            # 将驾驶员重新加入到节点中
                city.n_drivers += 1                                    # 更新城市中的可用驾驶员数量
            elif self.city_time < order_end_time:                      # 如果当前时间小于订单结束时间
                pass                 # 继续等待订单完成，什么都不做
            else:                   # 如果当前时间大于订单结束时间，出现错误
                raise ValueError('Driver: status_control_eachtime(): order end time less than city time')

# 表示一个 订单 对象，包含订单的起始位置、目的地、开始时间、持续时间、价格、等待时间等信息。它提供了多个方法来访问和操作这些属性。这些属性和方法可以用于模拟或调度系统中管理订单的生命周期（从创建到完成）
class Order(object):
    __slots__ = ('_begin_p', '_end_p', '_begin_t',
                 '_t', '_p', '_waiting_time', '_assigned_time')

    def __init__(self, begin_position, end_position, begin_time, duration, price, wait_time):
        self._begin_p = begin_position  # node
        self._end_p = end_position      # node
        self._begin_t = begin_time
        # self._end_t = end_time
        self._t = duration              # the duration of order.
        self._p = price
        self._waiting_time = wait_time  # a order can last for "wait_time" to be taken
        self._assigned_time = -1

        '''
        begin_position: 订单的起始位置（一个节点对象）。
        end_position: 订单的目标位置（一个节点对象）。
        begin_time: 订单的开始时间。
        duration: 订单的持续时间，表示订单需要多少时间来完成。
        price: 订单的价格。
        wait_time: 订单的等待时间，表示订单最多可以等待多长时间才会被接单。
        assigned_time: 订单被分配的时间，初始值为 -1，表示还未分配。
        '''


    def get_begin_position(self):
        return self._begin_p

    def get_begin_position_id(self):
        return self._begin_p.get_node_index()

    def get_end_position(self):
        return self._end_p

    def get_begin_time(self):
        return self._begin_t

    def set_assigned_time(self, city_time):
        self._assigned_time = city_time

    def get_assigned_time(self):
        return self._assigned_time

    # def get_end_time(self):
    #     return self._end_t

    def get_duration(self):
        return self._t

    def get_price(self):
        return self._p

    def get_wait_time(self):
        return self._waiting_time
