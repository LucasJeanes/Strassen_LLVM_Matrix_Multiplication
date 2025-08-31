import numpy as np # type: ignore
import time

A = np.array([[1,2],[3,4]], dtype=np.int32)
B = np.array([[5,6],[7,8]], dtype=np.int32)

start = time.perf_counter()
for _ in range(100000):
    C = A.dot(B)
end = time.perf_counter()

print(C)
print("Time:", end-start)
