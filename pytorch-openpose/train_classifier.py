import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 加载特征数据
data = np.load('features_labels.npy', allow_pickle=True).item()
X = data['X']
y = data['y']

print("原始类别分布:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

# 合并左右侧倾
y_merged = y.copy()
y_merged[y_merged == 'Left Leaning'] = 'Leaning'
y_merged[y_merged == 'Right Leaning'] = 'Leaning'

print("\n合并后类别分布:")
unique, counts = np.unique(y_merged, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y_merged, test_size=0.2, random_state=42, stratify=y_merged
)

print(f"\n训练集: {len(X_train)} 个样本")
print(f"测试集: {len(X_test)} 个样本\n")

# 训练
print("开始训练...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_train, y_train)
print("训练完成!\n")

# 评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"测试集准确率: {accuracy*100:.2f}%\n")
print("详细分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=150)
print("\n混淆矩阵已保存到 confusion_matrix_final.png")

# 保存模型
joblib.dump(clf, 'posture_classifier_final.pkl')
print("最终模型已保存到 posture_classifier_final.pkl")