import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# データの読み込み
df = pd.read_csv("../wine.csv")

# 説明変数(X)と目的変数(y)
X = df.drop("quality", axis=1)
y = df["quality"]

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 線形回帰モデル
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== 線形回帰結果 ===")
print(f"平均二乗誤差(MSE)：{mse:.3f}")
print(f"決定係数(R²)：{r2:.3f}")

print("\n各説明変数の係数")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name:25s}: {coef:.4f}")

print(f"\n切片：{model.intercept_:.4f}")

# ------------------------
# グラフ作成
# ------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)

# 理想線
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression Result")

# wine-analysis直下に保存
plt.savefig("../linear_regression_result.png")

plt.show()
