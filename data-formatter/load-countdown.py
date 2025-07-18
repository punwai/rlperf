import json
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

TRAIN_SET_SIZE = 10000
TEST_SET_SIZE = 1000

def get_prompt(target: int, nums: list[int]) -> str:
    if not isinstance(target, int):
        raise ValueError("Target must be an integer")
    if not isinstance(nums, list) or not all(isinstance(num, int) for num in nums):
        raise ValueError("Nums must be a list of integers")
    
    user_prompt = f"""
You are an expert at the game of Countdown. The rules are as follows:
1. You will be given a target number and a list of input numbers.
2. You will need to form an arithmetic operations with the input numbers such that the result is the target number. You
must use each input number exactly once. If you do not use all the input numbers, your answer is incorrect. If you use
the same number twice, your answer is incorrect. If you use a number that is not in the input list, your answer is incorrect.
3. You must return the solution in the format <answer>arithmetic expression</answer>, where your answer contains the left-hand side of the equation.
4. You must contain your reasoning in the format <think>your reasoning here</think>. Your answer must be after the closing </think> tag.

Example Task:
Input numbers: [1, 2, 3, 4]
Target number: 10

# Correct answer:
<answer>(1 * 2 * 3) + 4</answer>

Wrong answers:
<answer>(1 * 2 * 3) + 4</answer> # This is incorrect because you did not only return the left-hand side of the equation.
<answer>(2 * 3) + 4</answer> # This is incorrect because you did not use all the input numbers.
<answer>(1 * 1 * 2 * 3) + 4</answer> # This is incorrect because you used the number 1 twice.

-------------------------------------------
Your task is to solve the following countdown task:

Input numbers: {nums}
Target number: {target}
"""
    return json.dumps([{
        "role": "user",
        "content": user_prompt,
    }])


# The Dataloader should load data from a local .parquet file.
train_loader: Dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
train_loader = train_loader.select(range(TRAIN_SET_SIZE + TEST_SET_SIZE))
train_loader = train_loader.map(lambda x: {"prompt": get_prompt(x["target"], x["nums"])})

test_loader = train_loader.select(range(TRAIN_SET_SIZE, TRAIN_SET_SIZE + TEST_SET_SIZE))

train_loader.to_parquet("data/countdown/train.parquet")
test_loader.to_parquet("data/countdown/test.parquet")
