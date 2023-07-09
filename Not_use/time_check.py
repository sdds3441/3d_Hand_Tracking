import time
import math

start = time.time()

list=['1','2','2','3','2']
# while start + 5 > time.time():
#     print(f"{time.time() - start:.5f} sec")
print(max(set(list),key=list.count))
# while time.time()
# end = time.time()
#
# if
# print(f"{end - start:.5f} sec")
