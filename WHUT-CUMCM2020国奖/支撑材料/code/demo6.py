import numpy as np
from maze import get_maze
import random
import matplotlib.pyplot as plt
import seaborn as sns

jilu_jiage = []

# ISREAD_MAZES = True

guanqia = 6 #关卡
learn_num = 100000  #迭代次数
W_UP = 1200    #负重上线
aleph, gamma = 0.8, 0.8 #学习率
shouyi = 1000   # 挖矿收益
ddd = 30    # 天数
cunzhuang = [14]
kuang = [18]        # 矿山
maze, _ = get_maze(guanqia)
wakuang = maze.shape[0]  # 终点位置
# PATH = 'demo4'
starts_path= [1]
starts_A= []
start_t = len(starts_path) - 1
SSSSS = starts_path[len(starts_path) - 1]# 起点
# gaowengailv = 0.3 + 0.05 # 高温概率(其中0.05是沙暴概率，固定）
shabao = 0.05


def get_weather(gaowengailv):
    weatherssss = []
    for i in range(ddd):
        x = random.random()
        if x<shabao:
            weatherssss.append(3)
        elif x>=shabao and x<gaowengailv:
            weatherssss.append(2)
        else:
            weatherssss.append(1)
    return weatherssss


def get_w_i(weather):
    w_i, p_i = 0, 0
    if weather==1:
        w_i = 3*3+2*4
        p_i = 3*5+10*4
    elif weather==2:
        w_i = 9*3+2*9
        p_i = 9*5+10*9
    else:
        w_i = 3*10+2*10
        p_i = 5*10+10*10
    return w_i, p_i


def get_pathmap(maze):
    path_map = []
    # 每个状态拥有的动作
    for i in range(maze.shape[0]):  # 初始化所有可行域
        temp = []
        for j in range(1, maze.shape[1]):
            if not np.isnan(maze[i,j]):
                temp.append(j)
        path_map.append(temp)
    return path_map


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


def get_W_status(starts_path, starts_A, weathers):
    s = 1  # 起点
    w = 0  # 背包负重
    w_chunzhuang = 0  # 村庄买的东西
    plan = [W_UP, 0]  # 计划购买栈
    Rs = []
    flag = False    # 是否饿死
    for t in range(len(starts_path)-1):# 第0天就开始动，第30天没有动
        where = starts_path[t]
        s_p = starts_A[t] if starts_A[t] != wakuang else where#下一步要干嘛

        w_i, r_i = get_w_i(weathers[t])  # 第t天的基础消耗
        R = -r_i  # 基础消耗
        w_i_i = w_i # 基础消耗
        if where!=s_p:  # 走，2倍
            R = - 2 * r_i
            w_i_i = 2 * w_i
        if s_p == wakuang:  # 挖矿钱
            R = -3 * r_i + shouyi
            w_i_i = 3 * w_i
        w = w + w_i_i
        w_chunzhuang = w_chunzhuang + w_i_i

        if plan[0] >= w_i_i:  # 不需要村庄
            plan[0] = plan[0] - w_i_i
        else:  # 需要用村庄
            plan[1] = plan[1] - w_i_i + plan[0]  # 村庄装不够的
            plan[0] = 0
            if where!=s_p:  # 走，2倍
                R = - 4 * r_i
            if s_p == wakuang:  # 挖矿钱
                R = -6 * r_i + shouyi  # 挖矿钱
            if plan[1] < 0:
                flag = True # 死了

        print("第{}天执行{}->{}".format(t,where,s_p), plan, w_chunzhuang,w)

        # 村庄买东西
        if where in cunzhuang:  # 村庄
            plan[1] = plan[1] + w_chunzhuang
            w_chunzhuang = 0

        Rs.append(R)
    return plan,Rs,flag


def get_path(mazes, path_map, weathers):
    path = starts_path.copy()
    A = starts_A.copy()
    s = SSSSS
    for t in range(start_t, len(weathers)-1):  # (10, 30)
        map = mazes[t,:,:]
        if s==wakuang - 1:
            break
        a = probability_fun(path_map[s], map[s,:], 0)
        s_p = a if a!=wakuang else s

        path.append(s_p)
        A.append(a)
        s = s_p
    return path, A


def get_x_y(weather):
    if weather==1:
        x = 3
        y = 4
    elif weather==2:
        x = 9
        y = 9
    else:
        x = 10
        y = 10
    return x, y


def get_R_x_ys(path,weather,sss):
    zhuan = 0
    for i in path:
        if i==sss:
            zhuan = zhuan+1000
    xs,ys=[],[]
    s=1
    for t in range(1,len(path)):
        a,b = get_x_y(weather[t-1])
        s_p = path[t]
        # s_p=get_index(s,a_pp)
        A=a
        B=b
        if s_p != s and s_p!=sss:  # 走，2倍
            A=2*a
            B=2*b
        if s_p == sss:
            A = 3 * a
            B = 3 * b
        xs.append(A)
        ys.append(B)
        # print(s,"->>",s_p,"   ", A,B)
        s = s_p

    x=sum(xs)
    y=sum(ys)
    R = int((1200 - 2 * y)/3) * 5 + (x + y - int((1200 - 2 * y)/3)) * 10 - zhuan
    return -R,xs,ys,zhuan


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
        # print(t,s,plan, w_chunzhuang,w)

        # 村庄买东西
        if s in cunzhuang:  # 村庄
            plan[1] = plan[1] + w_chunzhuang
            w_chunzhuang = 0

        s = path[t]
        Rs.append(R)
    return plan,Rs,flag


def main(gaowengailv, ISREAD_MAZES, PATH):
    mazes = []      # 时间图Q
    weathers = get_weather(gaowengailv)
    for i in range(len(weathers)):
        mazes.append(maze)
    mazes = np.array(mazes).copy()
    path_map = get_pathmap(maze)
    # print(mazes.shape)  # mazes (时间，状态，动作）
    # print(path_map)
    # print(path_map[21])

    #   mazes付初值
    if ISREAD_MAZES:
        mazes = np.array(np.load("./data/{}_{}.npz".format(PATH,guanqia))['data'])
    # 初值的可视化
    #     cmap = sns.cubehelix_palette(start=3, rot=4, as_cmap=True, dark=0.7, light=0.3)
    #     sns.heatmap(mazes[0,:,:], linewidths=0.05, cbar=True, cmap=cmap)    #
    #     plt.show()
    mazes_demo = mazes.copy()

    print('######## Q-learing ########')
    plansss, Rs, flag = get_W_status(starts_path, starts_A, weathers)
    # print(flag, plansss)
    print("起点", SSSSS, "第几天", start_t, len(weathers)-1,"矿位置", wakuang)

    huos = 0
    jiage = []
    for k in range(learn_num):# 迭代次数
        s = SSSSS  # 起点

        plan = plansss.copy()   # 计划购买栈
        path = starts_path.copy()
        w = 1200-plansss[0]  # 背包负重
        w_chunzhuang = 1200-plansss[0]    # 村庄买的东西
        A = starts_A.copy()
        flag = False
        huo = 0
        # print("*"*50)
        # is_kuang = False
        # print(start_t, len(weathers))
        if ISREAD_MAZES:
            mazes = mazes_demo.copy()
            weathers = get_weather(gaowengailv).copy()

        for t in range(start_t, len(weathers)-1):#(10, 30)

            a = probability_fun(path_map[s], mazes[t, s, :], k=k)  # 动作
            # if s not in kuang and a in kuang:
            #     is_kuang=True
            # if is_kuang:
            #     a=s
            #     is_kuang = False

            s_p = a if a != wakuang else s  # 下一步要干嘛

            w_i, r_i = get_w_i(weathers[t])
            R = -r_i  # 走一步消耗
            w_i_i = w_i
            if a != s:  # 走，2倍
                R = - 2 * r_i
                w_i_i = 2 * w_i
            if a == wakuang:
                R = -3 * r_i + shouyi  # 挖矿钱
                w_i_i = 3 * w_i
            if weathers[t] == 3:  # 沙包必须停留
                s_p = s
                # if a!=wakuang and a!=s:
                #     a=s

            w = w + w_i_i
            w_chunzhuang = w_chunzhuang + w_i_i

            if plan[0] >= w_i_i:  # 不需要村庄
                plan[0] = plan[0] - w_i_i
            else:  # 需要用村庄
                plan[1] = plan[1] - w_i_i + plan[0]  # 村庄装不够的
                plan[0] = 0
                if s != s_p:  # 走，2倍
                    R = - 4 * r_i
                if a == wakuang:  # 挖矿钱
                    R = -6 * r_i + shouyi  # 挖矿钱
                if plan[1] < 0:
                    flag = True  # 死了
            if plan[1] < 0:
                flag = True  # 死了

            # print("第{}天执行{}, {}->{}".format(t,a,s,s_p), plan)

            # 村庄买东西
            if s in cunzhuang:  # 村庄
                plan[1] = plan[1] + w_chunzhuang
                w_chunzhuang = 0

            max_q = mazes[t + 1, s_p, probability_fun(path_map[s_p], mazes[t + 1, s_p, :], 0)]  # 找到最大位置的q值

            if s_p == wakuang - 1:  # 终点
                huo = 1
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * R
                s = s_p  # 更新位置
                # print(path)
                break       # 29
            elif flag:  # 中止条件
                R = R - 1000000
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma * max_q)
                s = s_p  # 更新位置
                break
            elif t == len(weathers) - 1 -1 and s_p != wakuang-1:
                R = R - 1000000
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma * max_q)
                s = s_p  # 更新位置
                break
            else:
                mazes[t, s, a] = (1 - aleph) * mazes[t, s, a] + aleph * (R + gamma * max_q)
                s = s_p  # 更新位置
            A.append(a)
            path.append(s)
        # break


        A.append(25)
        pathsss = [1]
        for i in A:
            pathsss.append(i)
        if huo:
            huos = huos + 1  # 活着
            R, xs, ys, zhuan = get_R_x_ys(pathsss, weathers, 26)
            # print(R)
            # print(pathsss)
            jiage.append(R)

    if ISREAD_MAZES:
        assert len(jiage)==huos
        print("平均收益", sum(jiage)/huos, '活着概率', round(huos*100/learn_num,2),"%")
    else:
        np.savez_compressed("./data/{}_{}.npz".format(PATH,guanqia), data=mazes)

        path, A = get_path(mazes, path_map, weathers)
        # print(path)
        # print(A)
        paths = [1]
        for i in A:
            paths.append(i)

        # print(len(paths))
        plan,Rs, flag = get_awser(maze, paths, weathers)
        print(paths)
        # print(plan)
        # print(Rs)
        print("****************************")
        print(flag)
        R, xs, ys, zhuan = get_R_x_ys(paths, weathers, 26)
        print(R)


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


if __name__ == '__main__':

    # for gaowengailv in np.arange(0,1-shabao,0.1):
    #     ISREAD_MAZES = False
    #     yuzhi = int(learn_num * 0.9) if not ISREAD_MAZES else 0  # 策略域
    #
    #     gaowengailv2 = gaowengailv+shabao
    #     main(gaowengailv2, ISREAD_MAZES, 'demo6_gailv_2_01_{}'.format(int(gaowengailv * 100)))
    #
    #     ISREAD_MAZES = True
    #     yuzhi = int(learn_num * 0.9) if not ISREAD_MAZES else 0  # 策略域
    #     main(gaowengailv2, ISREAD_MAZES, 'demo6_gailv_2_01_{}'.format(int(gaowengailv * 100)))


    huozhe  = np.array([88.05,61.2,25.03,64.89,34.8,
               30.23,80.05 ,12.1 ,21.5, 87.07])
    jiage = np.array([10461.8539,13351.1500,9176.8173,5457.4128,5252.251,
             4217.9555, 1592.151,3037.4466,399.0921,111.672])/3

    huozhe2  = np.array([64.15,25.89,4.66,44.96,57.35 ,
                29.9,  35.93,38.49,96.08, 14.27,])
    jiage2 =np.array( [12210.633,8490.3822,11078.8149, 6299.4749, 4370.22,
              4201.05424,3225.40552,2148.689,-208.6307,407.588972])/3

    huozhe3  = np.array([26.29 , 91.12 ,24.6,8.31,6.87  ,
                88.73,  35.9,38.0,50.37, 87.23,])
    jiage3 = np.array([14165.0 ,2747.553, 9173.948, 6299.4749,  7124.87,
              2045.927, 3196.399,2165.566, 1098.84858,100.521])/3

    huozhe = (huozhe+huozhe2+huozhe3)/3
    x = np.arange(0,0.95,0.1)+shabao
    plt.plot(x, huozhe, label="平均存活率", linestyle='--', marker='*')
    plt.ylim((0,100))
    plt.show()



    x = np.arange(0,0.95,0.1)+shabao
    fig, ax1 = plt.subplots()
    ax1.plot(x, huozhe, label="agent1存活率", linestyle='--', marker='*')
    ax1.plot(x, huozhe2, label="agent2存活率", linestyle='--', marker='*')
    ax1.plot(x, huozhe3, label="agent3存活率", linestyle='--', marker='*')
    ax1.plot([1], [0], linestyle=':')
    plt.grid(linestyle='-.')
    ax1.set_xlabel("高温概率")
    ax1.set_ylabel("存活率/%")
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

    fig, ax2 = plt.subplots()  # 做镜像处理
    ax2.plot(x, jiage, label="agent1平均保留资金")
    ax2.plot(x, jiage2, '-r',label="agent2平均保留资金")
    ax2.plot(x, jiage3, '-k',label="agent3平均保留资金")
    plt.grid(linestyle='-.')
    ax2.set_ylabel("平均保留资金")
    ax2.set_xlabel("高温概率")
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # plt.savefig(r'C:\Users\77526\Desktop\CUMCM\precode\pic\iteration.png')
    plt.show()