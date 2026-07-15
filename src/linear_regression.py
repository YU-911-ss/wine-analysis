import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# データの読み込み
df = pd.read_csv("data/wine.csv")

# 説明変数(X)と目的変数(y)
X = df[["alcohol"]]     # 説明変数
y = df["quality"]       # 目的変数

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
print(f"回帰式：quality = {model.coef_[0]:.3f} × alcohol + {model.intercept_:.3f}")
print(f"平均二乗誤差(MSE)：{mse:.3f}")
print(f"決定係数(R²)：{r2:.3f}")

# -----------------------------
# グラフ作成
# -----------------------------

plt.figure(figsize=(8,6))

# 散布図
plt.scatter(X_test["alcohol"], y_test,
            color="blue", alpha=0.7, label="Actual Data")

# 回帰直線
X_sorted = X_test.sort_values(by="alcohol")
y_line = model.predict(X_sorted)

plt.plot(X_sorted["alcohol"], y_line,
         color="red", linewidth=2, label="Regression Line")

plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression (Alcohol vs Quality)")
plt.legend()

# wine-analysis直下に保存
plt.savefig("linear_regression_result.png")

plt.show()
