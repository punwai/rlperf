# import ray
# from rlperf.rollout_engine import RolloutEngine

# 
# 1. Get the rollout engine to work for single-node GRPO RL.
# 

# def main():
#     ray.init()

#     engine = RolloutEngine.remote(
#         model_name="Qwen/Qwen3-0.6B",
#         num_workers=1,
#     )
#     ray.get(engine.start_engine.remote())

# if __name__ == "__main__":
#     main()