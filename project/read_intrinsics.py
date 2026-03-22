import msgpack
import numpy as np

with open("world.intrinsics", "rb") as fh:
    camera_data = msgpack.unpack(fh, raw=False)

# find resolution key automatically
res_key = None
for k in camera_data.keys():
    if k != "version":
        res_key = k
        break

print("Using resolution key:", res_key, type(res_key))

K = np.array(camera_data[res_key]["camera_matrix"])
dist = np.array(camera_data[res_key]["dist_coefs"])

print("K:\n", K)
print("dist:\n", dist)
