import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

# データの読み込み
df = pd.read_csv("data/wine.csv")

# 説明変数(X)と目的変数(y)
X = df[["alcohol"]]
y = df["quality"]

# 線形回帰
model = LinearRegression()
model.fit(X, y)

# 回帰係数
a = model.coef_[0]
b = model.intercept_

print(f"回帰式：quality = {a:.3f} × alcohol + {b:.3f}")

# 散布図
plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", alpha=0.6, label="Data")

# 回帰直線
x_line = np.linspace(X["alcohol"].min(), X["alcohol"].max(), 100)
y_line = a * x_line + b

plt.plot(x_line, y_line, color="red", linewidth=2,
         label=f"y = {a:.3f}x + {b:.3f}")

plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression")
plt.legend()

# 画像保存
plt.savefig("linear_regression_result.png")

plt.show()
