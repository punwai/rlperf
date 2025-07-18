print("Starting pynvml test")
import time
start_time = time.time()
import pynvml
end_time = time.time()
print(f"pynvml imported in {end_time - start_time} seconds.")


start_time = time.time()
pynvml.nvmlInit()
end_time = time.time()
print(f"pynvml.nvmlInit took {int((end_time - start_time) * 1000)} ms")

start_time = time.time()
device_count = pynvml.nvmlDeviceGetCount()
print(f"Device count: {device_count}")
end_time = time.time()
print(f"pynvml.nvmlDeviceGetCount took {int((end_time - start_time) * 1000)} ms")

pynvml.nvmlShutdown()