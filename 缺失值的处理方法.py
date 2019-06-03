from sklearn.preprocessing import Imputer

import numpy as np


def imputer01():
     imp = Imputer(missing_values='NaN', strategy="mean", axis=0)
     data = imp.fit_transform([[1, 2],
                               [np.nan, 3], [7, 6]])
     print(data)



if __name__ == "__main__":
    imputer01()