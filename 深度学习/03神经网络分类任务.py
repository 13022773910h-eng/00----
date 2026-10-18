import torch
#print(torch.__version__)

from pathlib import Path
import requests
import pickle
import gzip

# ========== 1. 加载MNIST数据集（这里用简单方式拿到x_train,y_train,x_valid,y_valid）
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

# 如果文件不存在就下载
if not (PATH / FILENAME).exists():
    try:
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
        print("下载完成")
    except Exception as e:
        print("下载失败！国内访问deeplearning.net不稳定，建议手动下载")
        print(f"错误信息：{e}")

# ========== 重点：Python3.10读取pkl.gz ==========
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    # 加上 encoding="latin-1" 兼容老版本pickle
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#print("x_train shape:", x_train.shape)
#print("y_train shape:", y_train.shape)

# ========== 2. 可视化第一张图片（就是刚才看到的数字5）
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
#print(x_train.shape)
pyplot.show()   # 加上这一行！

# ========== 3. numpy数组转tensor，就是你截图里核心代码！
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

n, c = x_train.shape
print("\n==== tensor信息 ====")
print("x_train tensor:\n", x_train)
print("x_train shape:", x_train.shape)
print("y_train min:", y_train.min(), "y_train max:", y_train.max())