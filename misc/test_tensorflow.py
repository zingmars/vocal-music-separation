# Test whether tensorflow has picked up our GPU.
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
