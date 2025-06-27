import numpy as np
import os
import errno



from datetime import datetime, timedelta

# 从 start 到 end，每隔一个时间间隔 delta，依次产出一个时间点
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current  #yield 会让函数变成一个生成器，每次调用时返回一个值，并记住当前状态，下次从上一次 yield 的地方继续执行。
        current += delta

# 安全地创建多层目录
# 定义一个函数，接收一个路径字符串 path，用于创建目录。
def mkdir_p(path):
    try:
        os.makedirs(path) #尝试创建目录（包括中间的所有父目录），如果目录已存在，会抛出异常。
    except OSError as exc:  # Python >2.5
        # 如果抛出的错误是 "文件已存在"（errno.EEXIST），并且这个路径确实是个目录（不是文件），那就 忽略这个错误，程序继续执行。
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# mkdir_p("/content/ds")

# 将二维矩阵中的坐标 (i, j) 转换为一维列表中的索引。
def ids_2dto1d(i, j, M, N):
    '''
    convert (i,j) in a M by N matrix to index in M*N list. (row wise)
    matrix: [[1,2,3], [4, 5, 6]]
    list: [0, 1, 2, 3, 4, 5, 6]
    index start from 0
    '''
    assert 0 <= i < M and 0 <= j < N # 保证索引合法
    index = i * N + j   # 计算一维索引
    return index   # 返回结果

# 将一维列表中的索引转换为二维矩阵中的坐标 (i, j)
def ids_1dto2d(ids, M, N):
    ''' inverse of ids_2dto1d(i, j, M, N)
        index start from 0
    '''
    i = ids // N # 整除，得到行索引
    j = ids - N * i
    return (i, j)

ids_2dto1d(3,4,10,10), ids_1dto2d(34,10,10)

# 根据给定的二维地图坐标 (i, j)，返回该点的 相邻节点列表，支持两种网格类型：
# 六边形网格（n=6，模拟真实城市中街区的蜂窝状分布）
# 四边形网格（n=4，常规棋盘式网格）
def get_neighbor_list(i, j, M, N, n, nodes):
    ''' n: n-sided polygon, construct for a 2d map
                 1
             6       2
               center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    '''

    neighbor_list = [None] * n
    if n == 6:
        # hexagonal
        if j % 2 == 0:
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]
            if j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i,   j+1, M, N)]
            if i + 1 < M and j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i+1, j+1, M, N)]
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i+1, j,   M, N)]
            if i + 1 < M and j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i+1, j-1, M, N)]
            if j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i,   j-1, M, N)]
        elif j % 2 == 1:
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]
            if i - 1 >= 0 and j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i-1, j+1, M, N)]
            if j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i,   j+1, M, N)]
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i+1, j,   M, N)]
            if j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i,   j-1, M, N)]
            if i - 1 >= 0 and j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i-1, j-1, M, N)]
    elif n == 4:
        # square
        if i - 1 >= 0:
            neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]
        if j + 1 < N:
            neighbor_list[1] = nodes[ids_2dto1d(i,   j+1, M, N)]
        if i + 1 < M:
            neighbor_list[2] = nodes[ids_2dto1d(i+1, j,   M, N)]
        if j - 1 >= 0:
            neighbor_list[3] = nodes[ids_2dto1d(i,   j-1, M, N)]

    return neighbor_list

'''
#get_neighbor_list(i, j, M, N, n, nodes)
nodes = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
]

get_neighbor_list(0, 0, 5, 5, 4, nodes)
'''

# 返回给定位置 (i, j) 在六边形网格中的六个邻居的索引。
def get_neighbor_index(i, j):
    """
                 1
             6       2
                center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    """
    neighbor_matrix_ids = []
    if j % 2 == 0:
        neighbor_matrix_ids = [[i - 1, j    ],
                               [i,     j + 1],
                               [i + 1, j + 1],
                               [i + 1, j    ],
                               [i + 1, j - 1],
                               [i    , j - 1]]
    elif j % 2 == 1:
        neighbor_matrix_ids = [[i - 1, j    ],
                               [i - 1, j + 1],
                               [i    , j + 1],
                               [i + 1, j    ],
                               [i    , j - 1],
                               [i - 1, j - 1]]

    return neighbor_matrix_ids

get_neighbor_index(0,0)

# 获取一个中心节点 (i, j) 在六边形网格上，按层级（层数）依次展开的邻居节点
def get_layers_neighbors(i, j, l_max, M, N):
    """get neighbors of node layer by layer, todo BFS.
       i, j: center node location
       L_max: max number of layers
       layers_neighbors: layers_neighbors[0] first layer neighbor: 6 nodes: can arrived in 1 time step.
       layers_neighbors[1]: 2nd layer nodes id
       M, N: matrix rows and columns.
    """
    assert l_max >= 1
    layers_neighbors = []
    layer1_neighbor = get_neighbor_index(i, j)  #[[1,1], [0, 1], ...]
    temp = []
    for item in layer1_neighbor:
        x, y = item
        if 0 <= x <= M-1 and 0 <= y <= N-1:
            temp.append(item)
    layers_neighbors.append(temp)

    node_id_neighbors = []
    for item in layer1_neighbor:
        x, y = item
        if 0 <= x <= M-1 and 0 <= y <= N-1:
            node_id_neighbors.append(ids_2dto1d(x, y, M, N))

    layers_neighbors_set = set(node_id_neighbors)
    curr_ndoe_id = ids_2dto1d(i, j, M, N)
    layers_neighbors_set.add(curr_ndoe_id)

    t = 1
    while t < l_max:
        t += 1
        layer_neighbor_temp = []
        for item in layers_neighbors[-1]:
            x, y = item
            if 0 <= x <= M-1 and 0 <= y <= N-1:
                layer_neighbor_temp += get_neighbor_index(x, y)

        layer_neighbor = []  # remove previous layer neighbors
        for item in layer_neighbor_temp:
            x, y = item
            if 0 <= x <= M-1 and 0 <= y <= N-1:
                node_id = ids_2dto1d(x, y, M, N)
                if node_id not in layers_neighbors_set:
                    layer_neighbor.append(item)
                    layers_neighbors_set.add(node_id)
        layers_neighbors.append(layer_neighbor)

    return layers_neighbors

# get_layers_neighbors(2, 2, 3, 5, 5)

# 用于获取网格中每个节点上闲置驾驶员的分布情况。函数会遍历所有的驾驶员，检查每个驾驶员的状态（是否闲置且在线），并记录其所在节点的闲置驾驶员数量。最终返回一个矩阵，表示网格中每个节点上闲置驾驶员的数量。
def get_driver_status(env):
    idle_driver_dist = np.zeros((env.M, env.N))
    for driver_id, cur_drivers in env.drivers.items():
        if cur_drivers.node is not None:
            node_id = cur_drivers.node.get_node_index()
            row, col = ids_1dto2d(node_id, env.M, env.N)
            if cur_drivers.onservice is False and cur_drivers.online is True:
                idle_driver_dist[row, col] += 1

    return idle_driver_dist

# 用于调试打印某个节点上所有驾驶员的状态信息。它显示驾驶员的 ID、位置、是否在线、是否在服务中等信息，以便开发者可以检查和调试节点上的驾驶员状态。
def debug_print_drivers(node):
    print("Status of all drivers in the node {}".format(node.get_node_index()))
    print("|{:12}|{:12}|{:12}|{:12}|".format("driver id", "driver location", "online", "onservice"))

    for driver_id, cur_drivers in node.drivers.items():
        if cur_drivers.node is not None:
            node_id = cur_drivers.node.get_node_index()
        else:
            node_id = "none"
        print("|{:12}|{:12}|{:12}|{:12}|".format(driver_id, node_id, cur_drivers.online, cur_drivers.onservice))
