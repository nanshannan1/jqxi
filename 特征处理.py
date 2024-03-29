# 应用与传统精确小数据的场景
from sklearn.preprocessing import MinMaxScaler


def mm():
    '''
    归一化处理
    :return:
    '''
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90, 2, 10, 40],
                      [60, 4, 15, 45],
                      [75, 3, 13, 46]])
    print(data)


if __name__ == "__main__":
    mm()