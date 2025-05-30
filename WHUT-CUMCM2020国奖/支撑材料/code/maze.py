import numpy as np
from numpy import isnan
import matplotlib.pyplot as plt
import seaborn as sns

def get_maze(checkpointnum):
    """
    :param checkpointnum: 关卡数目
    :return: 迷宫，天气
    """
    maze = []
    weather = []    # 1晴天，2高温，3沙暴
    assert checkpointnum in list(range(1,7))
    if checkpointnum==1:
        weather = [2,2,1,3,1,2,3,1,2,2,
                   3,2,1,2,2,2,3,3,2,2,
                   1,1,2,1,3,2,1,1,2,2]
        maze = np.full((27+1, 28+1), np.nan)    # 28为挖，+1为0的地方不要
        maze[12,28] = 0     # 挖矿
        for i in range(1, 27+1):        # 停一天
            maze[i, i] = 0
        # 临边
        maze[1, 25],maze[1, 2]=0,0
        maze[2, 1],maze[2, 3]=0,0
        maze[3, 2],maze[3, 25],maze[3, 4]=0,0,0
        maze[4, 3],maze[4, 25],maze[4, 24],maze[4, 5]=0,0,0,0
        maze[5, 4],maze[5, 24],maze[5, 6]=0,0,0
        maze[6, 5],maze[6, 24],maze[6, 23],maze[6, 7]=0,0,0,0
        maze[7, 6],maze[7, 22],maze[7, 8]=0,0,0
        maze[8, 7],maze[8, 22],maze[8, 9]=0,0,0
        maze[9, 8],maze[9, 22],maze[9, 21],maze[9, 17],maze[9, 16],maze[9, 15],maze[9, 10]=0,0,0,0,0,0,0
        maze[10, 9],maze[10, 15],maze[10, 13],maze[10, 11]=0,0,0,0
        maze[11, 10],maze[11, 13],maze[11, 12]=0,0,0
        maze[12, 11],maze[12, 13],maze[12, 14]=0,0,0
        maze[13, 10],maze[13, 15],maze[13, 14],maze[13, 12],maze[13, 11]=0,0,0,0,0
        maze[14, 15],maze[14, 16],maze[14, 12],maze[14, 13]=0,0,0,0
        maze[15, 10],maze[15, 9],maze[15, 16],maze[15, 14],maze[15, 13]=0,0,0,0,0
        maze[16, 14],maze[16, 15],maze[16, 9],maze[16, 17],maze[16, 18]=0,0,0,0,0
        maze[17, 9],maze[17, 21],maze[17, 18],maze[17, 16]=0,0,0,0
        maze[18, 17],maze[18, 20],maze[18, 19],maze[18, 16]=0,0,0,0
        maze[19, 18],maze[19, 20]=0,0
        maze[20, 21],maze[20, 18],maze[20, 19]=0,0,0
        maze[21, 27],maze[21, 23],maze[21, 22],maze[21, 9],maze[21, 17],maze[21, 20]=0,0,0,0,0,0
        maze[22, 7],maze[22, 23],maze[22, 21],maze[22, 9],maze[22, 8]=0,0,0,0,0
        maze[23, 24],maze[23, 26],maze[23, 21],maze[23, 22],maze[23, 6]=0,0,0,0,0
        maze[24, 5],maze[24, 6],maze[24, 23],maze[24, 26],maze[24, 25],maze[24, 4]=0,0,0,0,0,0
        maze[25, 1],maze[25, 24],maze[25, 3],maze[25, 4],maze[25, 26]=0,0,0,0,0
        maze[26, 25],maze[26, 24],maze[26, 23],maze[26, 27]=0,0,0,0
        maze[27, 26],maze[27, 21]=0,0
        pass
    elif checkpointnum==2:
        weather = [2, 2, 1, 3, 1, 2, 3, 1, 2, 2,
                   3, 2, 1, 2, 2, 2, 3, 3, 2, 2,
                   1, 1, 2, 1, 3, 2, 1, 1, 2, 2]
        maze = np.full((64 + 1, 65 + 1), np.nan)
        maze[30,65] = 0     # 挖矿
        maze[55,65] = 0     # 挖矿
        for i in range(1, 64+1):
            maze[i, i] = 0 # 停一天
            paishu = int((i-1)/8)+1 # 第几排
            tou = (paishu - 1) * 8 + 1  # 头
            wei = paishu * 8        # 尾
            if paishu==1:
                if i==tou:
                    maze[tou,tou+1],maze[tou,tou+8]=0,0
                elif i==wei:
                    maze[wei,wei-1],maze[wei,wei+7],maze[wei,wei+8]=0,0,0
                else:
                    maze[i,i-1],maze[i,i+1],maze[i,i+7],maze[i,i+8]=0,0,0,0
            if paishu==8:
                if i==tou:
                    maze[tou,tou-8],maze[tou,tou-7],maze[tou,tou+1]=0,0,0
                elif i==wei:
                    maze[wei,wei-1],maze[wei,wei-8]=0,0
                else:
                    maze[i,i-1],maze[i,i+1],maze[i,i-8],maze[i,i-7]=0,0,0,0
            if paishu in [2,4,6]:
                if i==tou:
                    maze[tou,tou-8],maze[tou,tou-7],maze[tou,tou+1],maze[tou,tou+8],maze[tou,tou+9]=0,0,0,0,0
                elif i==wei:
                    maze[wei,wei-1],maze[wei,wei-8],maze[wei,wei+8]=0,0,0
                else:
                    maze[i,i-1],maze[i,i+1],maze[i,i-8],maze[i,i-7],maze[i,i+8],maze[i,i+9]=0,0,0,0,0,0
            if paishu in [3,5,7]:
                if i==tou:
                    maze[tou,tou+1],maze[tou,tou-8],maze[tou,tou+8]=0,0,0
                elif i==wei:
                    maze[wei,wei-1],maze[wei,wei-9],maze[wei,wei-8],maze[wei,wei+8],maze[wei,wei+7]=0,0,0,0,0
                else:
                    maze[i,i-1],maze[i,i+1],maze[i,i-8],maze[i,i-9],maze[i,i+8],maze[i,i+7]=0,0,0,0,0,0
        pass
    elif checkpointnum==3:
        weather = []
        maze = np.full((13 + 1, 14 + 1), np.nan)
        maze[9, 14] = 0  # 挖矿
        for i in range(1, 13 + 1):
            maze[i, i] = 0 # 停一天
        maze[1, 2],maze[1, 5],maze[1, 4]=0,0,0
        maze[2, 1],maze[2, 4],maze[2, 3]=0,0,0
        maze[3, 2],maze[3, 4],maze[3, 9],maze[3, 8]=0,0,0,0
        maze[4, 1],maze[4, 2],maze[4, 3],maze[4, 7],maze[4, 6],maze[4, 5]=0,0,0,0,0,0
        maze[5, 1],maze[5, 4],maze[5, 6]=0,0,0
        maze[6, 5],maze[6, 4],maze[6, 7],maze[6, 12],maze[6, 13]=0,0,0,0,0
        maze[7, 4],maze[7, 11],maze[7, 12],maze[7, 6]=0,0,0,0
        maze[8, 3],maze[8, 9]=0,0
        maze[9, 3],maze[9, 8],maze[9, 11],maze[9, 10]=0,0,0,0
        maze[10, 9],maze[10, 11],maze[10, 13]=0,0,0
        maze[11, 9],maze[11, 10],maze[11, 13],maze[11, 12],maze[11, 7]=0,0,0,0,0
        maze[12, 7],maze[12, 11],maze[12, 13],maze[12, 6]=0,0,0,0
        maze[13, 6],maze[13, 12],maze[13, 11],maze[13, 10]=0,0,0,0
        pass
    elif checkpointnum==4:
        weather = []
        maze = np.full((25 + 1, 26 + 1), np.nan)
        maze[18, 26] = 0  # 挖矿
        for i in range(1, 25 + 1):
            maze[i, i] = 0 # 停一天
            if i==1:
                maze[i,i+1],maze[i,i+5]=0,0
            elif i==5:
                maze[i,i-1],maze[i,i+5]=0,0
            elif i==21:
                maze[i,i+1],maze[i,i-5]=0,0
            elif i==25:
                maze[i,i-1],maze[i,i-5]=0,0
            elif i in [2,3,4]:
                maze[i,i-1],maze[i,i+1],maze[i,i+5]=0,0,0
            elif i in [22,23,24]:
                maze[i,i-1],maze[i,i+1],maze[i,i-5]=0,0,0
            elif i in [6,11,16]:
                maze[i,i-5],maze[i,i+1],maze[i,i+5]=0,0,0
            elif i in [10,15,20]:
                maze[i,i-5],maze[i,i-1],maze[i,i+5]=0,0,0
            else:
                maze[i, i-1], maze[i, i + 1], maze[i, i-5], maze[i, i+5] = 0, 0, 0, 0
        pass
    elif checkpointnum==5:
        weather = [1,2,1,1,1,1,2,2,2,2,]
        maze = np.full((13 + 1, 14 + 1), np.nan)
        maze[9, 14] = 0  # 挖矿
        for i in range(1, 13 + 1):
            maze[i, i] = 0 # 停一天
        maze[1, 2],maze[1, 5],maze[1, 4]=0,0,0
        maze[2, 1],maze[2, 4],maze[2, 3]=0,0,0
        maze[3, 2],maze[3, 4],maze[3, 9],maze[3, 8]=0,0,0,0
        maze[4, 1],maze[4, 2],maze[4, 3],maze[4, 7],maze[4, 6],maze[4, 5]=0,0,0,0,0,0
        maze[5, 1],maze[5, 4],maze[5, 6]=0,0,0
        maze[6, 5],maze[6, 4],maze[6, 7],maze[6, 12],maze[6, 13]=0,0,0,0,0
        maze[7, 4],maze[7, 11],maze[7, 12],maze[7, 6]=0,0,0,0
        maze[8, 3],maze[8, 9]=0,0
        maze[9, 3],maze[9, 8],maze[9, 11],maze[9, 10]=0,0,0,0
        maze[10, 9],maze[10, 11],maze[10, 13]=0,0,0
        maze[11, 9],maze[11, 10],maze[11, 13],maze[11, 12],maze[11, 7]=0,0,0,0,0
        maze[12, 7],maze[12, 11],maze[12, 13],maze[12, 6]=0,0,0,0
        maze[13, 6],maze[13, 12],maze[13, 11],maze[13, 10]=0,0,0,0
        pass
    elif checkpointnum==6:
        weather = []
        maze = np.full((25 + 1, 26 + 1), np.nan)
        maze[18, 26] = 0  # 挖矿
        for i in range(1, 25 + 1):
            maze[i, i] = 0 # 停一天
            if i==1:
                maze[i,i+1],maze[i,i+5]=0,0
            elif i==5:
                maze[i,i-1],maze[i,i+5]=0,0
            elif i==21:
                maze[i,i+1],maze[i,i-5]=0,0
            elif i==25:
                maze[i,i-1],maze[i,i-5]=0,0
            elif i in [2,3,4]:
                maze[i,i-1],maze[i,i+1],maze[i,i+5]=0,0,0
            elif i in [22,23,24]:
                maze[i,i-1],maze[i,i+1],maze[i,i-5]=0,0,0
            elif i in [6,11,16]:
                maze[i,i-5],maze[i,i+1],maze[i,i+5]=0,0,0
            elif i in [10,15,20]:
                maze[i,i-5],maze[i,i-1],maze[i,i+5]=0,0,0
            else:
                maze[i, i-1], maze[i, i + 1], maze[i, i-5], maze[i, i+5] = 0, 0, 0, 0
        pass
    else:
        print("输入关卡数错误")
        return None
    return maze, np.array(weather)


def is_duicheng(maze, num):
    maze = maze[:num, :num]
    print(maze.shape)
    for i in range(num):
        for j in range(num):
            if maze[i,j]!=maze[j,i] and not isnan(maze[j,i]):
                print("不是对称",i,j)




if __name__ =="__main__":
    maze,weather = get_maze(1)
    print(maze.shape)
    print(maze)
    is_duicheng(maze,maze.shape[0])
    # https://blog.csdn.net/qq_42554007/article/details/82624418
    cmap = sns.cubehelix_palette(start=3, rot=4, as_cmap=True,dark=0.6, light=0.6)
    sns.heatmap(maze, linewidths=0.05, cbar = True, cmap=cmap)    #
    plt.show()

