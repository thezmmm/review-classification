import pandas as pd

t1 = pd.Series([13, 23, 33, 43, 53], index=["a", "b", "c", "d", "e"])
print(t1)
print(type(t1))
'''
a    13
b    23
c    33
d    43
e    53
dtype: int64
'''

# 通过索引直接取值
print(t1["d"])  # 43
# 通过位置取值(从0开始)
print(t1[3])  # 43

