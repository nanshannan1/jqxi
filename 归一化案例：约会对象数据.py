# 标准化
# 对于标准化在样本足够多的情况下稳定
from sklearn.preprocessing import StandardScaler


def standard01():
    sd = StandardScaler()
    data = sd.fit_transform([[1., -1., 3.],
                             [2., 4., 2.],
                             [4., 6., -1.]])
    print(data)
    print(sd.mean_)
    # print(sd.std_)


if __name__ == "__main__":
    standard01()