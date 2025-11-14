# 通过混淆矩阵判定当前分类器的效果，具体哪些类别容易混淆，从而决定进一步的改进方案
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 加载数据和模型
data = np.load('features_labels.npy', allow_pickle=True).item()
X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = joblib.load('posture_classifier.pkl')
y_pred = clf.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

# 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("混淆矩阵已保存到 confusion_matrix.png")

# 打印混淆情况
print("\n主要混淆情况:")
for i, true_label in enumerate(clf.classes_):
    for j, pred_label in enumerate(clf.classes_):
        if i != j and cm[i][j] > 2:
            print(f"  {true_label} 被误判为 {pred_label}: {cm[i][j]} 次")