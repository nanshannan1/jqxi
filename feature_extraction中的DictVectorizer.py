# coding:utf-8
# 导入相关的数据包
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import jieba


def dictvec():
    '''
    字典数据抽取
    :return:
    '''

    # 实例化DictVectorizer, 并将结果转换为数组
    dict = DictVectorizer(sparse=False)
    # 调用fit_transform方法输入数据并转换
    data = dict.fit_transform([{'city': '北京', 'temperature': 100},
                        {'city': '上海', 'temperature': 60},
                        {'city': '深圳', 'temperature': 30}
                        ])
    print(dict.get_feature_names())
    print(data)
    print(dict.inverse_transform(data))


def countvec():
    '''
    对文本进行特征值化
    单个字符不统计
    对于中文不支持特证的抽取
    对中文进行抽取的话需要首先对文本进行分词在进行抽取
    :return:
    '''
    cv = CountVectorizer()
    data = cv.fit_transform(['life is short, i like python', "life is too long, i dislike python"])
    print(data)
    print(cv.get_feature_names())
    # print(cv.inverse_transform(data))
    print(data.toarray())
    return None


def cutword():
    c1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    c2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    c3 = jieba.cut('如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。')
    # 转换为列表

    content1 = list(c1)
    content2 = list(c2)
    content3 = list(c3)

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def hanzivec():
    '''
    中文特征值化
    :return:
    '''
    c1, c2, c3 = cutword()
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())


def tiidfvec():
    '''
    中文特征值化
    :return:
    '''
    c1, c2, c3 = cutword()

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names())
    print(data.toarray())


if __name__ == "__main__":
    tiidfvec()