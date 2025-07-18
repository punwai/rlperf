import time
start_time = time.time()
import torch
end_time = time.time()
print(f"torch imported in {end_time - start_time} seconds.")

def main():
    print("Hello from rlperf!")

if __name__ == "__main__":
    main()
