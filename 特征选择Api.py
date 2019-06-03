from sklearn.feature_selection import VarianceThreshold


def var():
    '''
    特征选择-删除低方差的特征
    :return:
    '''

    # 实例化对象
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0, 2, 0, 3],
                              [0, 1, 4, 3],
                              [0, 1, 1, 3]])
    print(data)


if __name__ == "__main__":
    var()