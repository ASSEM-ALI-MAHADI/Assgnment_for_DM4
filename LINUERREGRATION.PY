import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- إنشاء بيانات صناعية ---
np.random.seed(0)
m = 100
X = 2 * np.random.rand(m, 1)                # ميزة واحدة
y = 4 + 3.5 * X.flatten() + np.random.randn(m) * 0.8  # نضيف ضوضاء

# --- إنشاء النموذج وتدريبه ---
model = LinearRegression()
model.fit(X, y)

# --- طباعة المعاملات ---
print("معامل الانحدار (slope):   ", model.coef_[0])
print("المقطع (intercept):       ", model.intercept_)
print("معامل التحديد R^2:       ", model.score(X, y))

# --- رسم البيانات وخط الانحدار ---
plt.scatter(X, y, label="البيانات")
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_line)
plt.plot(x_line, y_pred, color='red', label="خط الانحدار")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression باستخدام sklearn")
plt.show()
