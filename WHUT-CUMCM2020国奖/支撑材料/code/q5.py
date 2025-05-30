import numpy as np
from maze import get_maze
import random
import matplotlib.pyplot as plt
import seaborn as sns
guanqia = 5 #关卡
learn_num = 10000  #迭代次数
yuzhi = int(learn_num*0.9) # 策略域
W_UP = 1200    #负重上线
aleph, gamma = 1, 0.8 #学习率
shouyi = 200   # 挖矿收益
cunzhuang = [15]
ISREAD_MAZES = False


def get_w_i(weather):
    w_i, p_i = 0, 0
    if weather==1:
        w_i = 3*5+2*7
        p_i = 5*5+10*7
    elif weather==2:
        w_i = 3*8+2*6
        p_i = 5*8+10*6
    else:
        w_i = 3*10+2*10
        p_i = 5*10+10*10
    return w_i, p_i


def probability_fun(sss, max_a, probability=0.8, k=None):
    '''
    :param probability: 0为贪婪，1为随机策略，其他为sigema策略
    '''
    if probability==1:
        return random.choice(sss)
    elif probability==0:
        list_a = max_a.tolist()
        q_max = float('-inf')
        for i in sss:
            if list_a[i] > q_max and not np.isnan(list_a[i]):
                q_max=list_a[i]
        # if probability==0:
        #     print(q_max, list_a.index(q_max))
        return list_a.index(q_max)
    else:
        if k<yuzhi:
            probability=-(1-0.5)*k/yuzhi+1
        else:
            probability=0
        if np.random.random()<probability:
            return random.choice(sss)
        else:
            list_a = max_a.tolist()
            q_max = float('-inf')
            for i in sss:
                if list_a[i] > q_max and not np.isnan(list_a[i]):
                    q_max = list_a[i]
            # if probability==0:
            #     print(q_max, list_a.index(q_max))
            return list_a.index(q_max)


def get_index(s,a,map):
    if a!=map.shape[1]-1:
        return a
    else:
        return s


def get_path(mazes, weather, path_map):
    s=1
    path = []
    path.append(s)
    for t in range(len(weather)):
        map = mazes[t,:,:]
        if s==map.shape[0]-1:
            break
        a = probability_fun(path_map[s], map[s,:], 0)
        path.append(a)
        s_p = get_index(s, a, map)
        s = s_p
    return path


def get_pathmap(maze):
    path_map = []
    #每个状态拥有的动作
    for i in range(maze.shape[0] ):  # 初始化所有可行域
        temp = []
        for j in range(1, maze.shape[1]):
            if not np.isnan(maze[i,j]):
                temp.append(j)
        path_map.append(temp)
    return path_map


def init_mazes(mazes, path, weather):
    for k in range(10000):
        # 判断终止条件
        print("*" * 50)
        s = 1  # 起点
        w = 0  # 背包负重
        w_chunzhuang = 0  # 村庄买的东西
        plan = [W_UP, 0]  # 计划购买栈
        flag = False

        for t in range(1, len(path)):

            a = path[t]
            s_p=get_index(s, a, mazes[t,:,:]) # 动作执行完后状态

            w_i, r_i = get_w_i(weather[t])
            R = -r_i  # 走一步消耗
            w_i_i = w_i
            if a != s:  # 走，2倍
                R = - 2 * r_i
                w_i_i = 2 * w_i
            if a == maze.shape[1] - 1:
                R = -3 * r_i + shouyi  # 挖矿钱
                w_i_i = 3 * w_i

            w = w + w_i_i
            w_chunzhuang = w_chunzhuang + w_i_i

            ##################
            # 村庄买东西
            if plan[0] >= w_i_i:  # 不需要村庄
                plan[0] = plan[0] - w_i_i
                pass
            else:  # 需要用村庄
                plan[1] = plan[1] - w_i_i + plan[0]  # 村庄装不够的
                plan[0] = 0
                if path[t] != s:  # 走，2倍
                    R = - 4 * r_i
                if path[t] == maze.shape[1] - 1:
                    R = -6 * r_i + shouyi  # 挖矿钱
                if plan[1] < 0:
                    flag = True

            # 村庄买东西
            if s in cunzhuang:  # 村庄
                plan[1] = plan[1] + w_chunzhuang
                w_chunzhuang = 0
            ##################
            max_q = mazes[t + 1, s_p, probability_fun(path_map[s_p], mazes[t + 1, s_p, :], 0)]  # 找到最大位置的q值
            if 15 in path:
                print(path, max_q, plan, w, w_chunzhuang, R, )

            if s_p == lenth - 1:  # 终点
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * R
                s = s_p  # 更新位置
                break
            elif flag or (t == len(weather) - 1 - 1 and s_p != lenth - 1):  # 中止条件
                R = R - 1000000
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma * max_q)
                s = s_p  # 更新位置
                break
            else:
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma * max_q)
                s = s_p  # 更新位置
    np.savez_compressed("./data/guan_init{}.npz".format(guanqia), data=mazes)
    return  mazes


def get_awser(maze,path, weather):
    s = 1  # 起点
    w = 0  # 背包负重
    w_chunzhuang = 0  # 村庄买的东西
    plan = [W_UP, 0]  # 计划购买栈
    Rs = []
    flag = False
    for t in range(1,len(path)):
        w_i, r_i = get_w_i(weather[t-1])
        R = -r_i  # 走一步消耗
        w_i_i = w_i
        if path[t] != s:  # 走，2倍
            R = - 2 * r_i
            w_i_i = 2 * w_i
        if path[t] == maze.shape[1]-1:
            R = -3 * r_i + shouyi  # 挖矿钱
            w_i_i = 3 * w_i
        w = w + w_i_i
        w_chunzhuang = w_chunzhuang + w_i_i

        if plan[0] >= w_i_i:  # 不需要村庄
            plan[0] = plan[0] - w_i_i
            pass
        else:  # 需要用村庄
            plan[1] = plan[1] - w_i_i + plan[0]  # 村庄装不够的
            plan[0] = 0
            if path[t] != s:  # 走，2倍
                R = - 4 * r_i
            if path[t] == maze.shape[1]-1:
                R = -6 * r_i + shouyi  # 挖矿钱
        if plan[1] < 0:
            flag = True
        print(t,s,plan, w_chunzhuang,w)

        # 村庄买东西
        if s in cunzhuang:  # 村庄
            plan[1] = plan[1] + w_chunzhuang
            w_chunzhuang = 0

        s = path[t]
        Rs.append(R)
    return plan,Rs,flag


from get_R import get_R_x_y

if __name__ == '__main__':
    maze, weather = get_maze(guanqia)
    path = [1,4,6,13]
    print(len(path))
    plan,Rs, flag = get_awser(maze, path, weather)
    print(path)
    print(plan)
    print(Rs)
    print("****************************")
    print(flag)
    R,xs,ys = get_R_x_y(path, weather, 28)
    print(R)

if __name__ == '__main__ssss;':
    maze, weather = get_maze(guanqia)
    lenth = maze.shape[0]   #终点位置
    print(weather, lenth)
    mazes = []      # 时间图
    for i in range(len(weather)):
        mazes.append(maze)
    mazes = np.array(mazes)
    path_map = get_pathmap(maze)
    print(mazes.shape)  # mazes (时间，状态，动作）
    print(path_map)

    # R放入初值中
    for t in range(len(weather)):
        for j in range(lenth):  # 初始化所有可行域
            for a in path_map[j]:
                w_i, r_i = get_w_i(weather[t])
                R = -r_i
                if a != j:  # 走，2倍
                    R = - 2 * r_i
                if a == maze.shape[1] - 1:
                    R = -3 * r_i + shouyi  # 挖矿钱
                mazes[t,j,a] = R

    init_path = [1, 25, 24, 23, 23, 22, 9, 9, 15, 14, 12, 28, 28, 28, 28, 28, 12, 28, 28, 13, 15, 9, 21, 27]

    cmap = sns.cubehelix_palette(start=3, rot=4, as_cmap=True, dark=0.7, light=0.3)
    sns.heatmap(mazes[0,:,:], linewidths=0.05, cbar=True, cmap=cmap)    #
    plt.show()

    mazes = init_mazes(mazes,init_path,weather)
    print(mazes)
    # mazes = np.array(np.load("./data/guan_init{}.npz".format(guanqia))['data'])

    if ISREAD_MAZES:
        mazes = np.array(np.load("./data/guan{}.npz".format(guanqia))['data'])

    cmap = sns.cubehelix_palette(start=3, rot=4, as_cmap=True, dark=0.7, light=0.3)
    sns.heatmap(mazes[0,:,:], linewidths=0.05, cbar=True, cmap=cmap)    #
    plt.show()


    ##q-learing
    print('######## Q-learing ########')
    for k in range(learn_num):# 迭代次数
        # 判断终止条件
        print("*"*50)
        s = 1  # 起点
        w = 0  # 背包负重
        w_chunzhuang = 0    # 村庄买的东西
        plan = [W_UP, 0]   # 计划购买栈
        path = [s]
        flag = False

        for t in range(len(weather)-1):
            a = probability_fun(path_map[s], mazes[t,s,:], k=k) #动作
            s_p = get_index(s, a, mazes[t,:,:]) # 动作执行完后状态
            w_i, r_i = get_w_i(weather[t])
            R = -r_i               # 走一步消耗
            w_i_i = w_i
            if a != s:    # 走，2倍
                R = - 2 * r_i
                w_i_i = 2*w_i
            if a==maze.shape[1]-1:
                R = -3*r_i+shouyi        # 挖矿钱
                w_i_i = 3*w_i
            if weather[t]==3:   # 沙包必须停留
                s_p = s

            w = w + w_i_i
            w_chunzhuang = w_chunzhuang + w_i_i

            ##################
            # 村庄买东西
            if plan[0] >= w_i_i:  # 不需要村庄
                plan[0] = plan[0] - w_i_i
                pass
            else:  # 需要用村庄
                plan[1] = plan[1] - w_i_i + plan[0]  # 村庄装不够的
                plan[0] = 0
                if path[t] != s:  # 走，2倍
                    R = - 4 * r_i
                if path[t] == maze.shape[1] - 1:
                    R = -6 * r_i + shouyi  # 挖矿钱
                if plan[1] < 0:
                    flag = True

            # 村庄买东西
            if s in cunzhuang:  # 村庄
                plan[1] = plan[1] + w_chunzhuang
                w_chunzhuang = 0
            ##################

            max_q = mazes[t+1, s_p, probability_fun(path_map[s_p], mazes[t+1,s_p,:], 0)] # 找到最大位置的q值
            if 15 in path:
                print(path, max_q, plan, w, w_chunzhuang, R,)

            if s_p==lenth-1:# 终点
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * R
                s = s_p  # 更新位置
                break
            elif flag or (t==len(weather)-1-1 and s_p != lenth-1):#中止条件
                R = R - 1000000
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma*max_q)
                s = s_p  # 更新位置
                break
            else:
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma*max_q)
                s = s_p  # 更新位置
            # print(mazes[t,s,a])
            path.append(a)
        pass


    # 输出路径
    # print(mazes)
    np.savez_compressed("./data/guan{}.npz".format(guanqia), data=mazes)

    path = get_path(mazes, weather, path_map)
    print(path)

    #
    # for t in range(25, len(weather)):
    #     map = mazes[t,:,:]
    #     for i in range(map.shape[0]):
    #         for j in range(map.shape[1]):
    #             if map[i,j]<-100000 and not np.isnan(map[i,j]):
    #                 map[i, j] = -1000000
    #     cmap = sns.cubehelix_palette(start=3, rot=4, as_cmap=True, dark=0.7, light=0.3)
    #     sns.heatmap(map, linewidths=0.05, cbar=True, cmap=cmap)    #
    #     plt.show()



