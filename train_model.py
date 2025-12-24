# train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# 创建模型保存目录
os.makedirs("models", exist_ok=True)

print("正在读取数据...")
df = pd.read_csv("student_data_adjusted_rounded.csv")

print(f"数据量：{len(df)} 条")
print("正在进行特征编码...")

# 编码器
le_gender = LabelEncoder()
le_major = LabelEncoder()

df['性别编码'] = le_gender.fit_transform(df['性别'])  # 男=1, 女=0
df['专业编码'] = le_major.fit_transform(df['专业'])

# 特征列
features = ['性别编码', '专业编码', '每周学习时长（小时）', '上课出勤率',
            '期中考试分数', '作业完成率']
X = df[features]
y = df['期末考试分数']

# 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("正在训练 XGBoost 模型（请稍等 10 秒）...")
model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=80,
          verbose=100)

# 评估
pred = model.predict(X_val)
r2 = r2_score(y_val, pred)
mae = mean_absolute_error(y_val, pred)

print("\n" + "="*50)
print(f"模型训练完成！")
print(f"验证集 R²  = {r2:.4f}  （越接近1越好）")
print(f"平均绝对误差 = {mae:.2f} 分")
print("="*50)

# 保存模型和编码器
joblib.dump(model, "models/xgb_final_predictor.pkl")
joblib.dump(le_gender, "models/le_gender.pkl")
joblib.dump(le_major, "models/le_major.pkl")

print("模型已保存至 ./models/ 目录")
print("现在你可以运行：streamlit run app.py 启动系统了！")
