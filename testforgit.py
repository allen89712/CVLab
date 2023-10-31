import numpy as np

sample = np.random.normal(0, 1, 10000)

std = np.std(sample)

mean = np.mean(sample)

print("平均值為 : " , mean )

print( "標準差為 : " , std)

sample2 = np.random.normal(500, 4, 10000)

std2 = np.std(sample2)

mean2 = np.mean(sample2)

print("平均值為 : " , mean2 )

print( "標準差為 : " , std2)
