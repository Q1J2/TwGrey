import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_excel(r"D:\UCM\UCM100\100.xlsx")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

feature_importances= {'correlation': 0.162608193844962, 'homogeneity': 0.141395546949546, 'sum_entropy': 0.137348405199838,  'sum_mean': 0.132346215584954, 'mean': 0.132214868440393, 'Value_Mean': 0.130908515145888, 'Hue_Mean': 0.11330556893641, 'dissimilarity': 0.109575841495347, 'mutual_information': 0.108883072240747, 'diff_entropy': 0.105572455036284, 'diff_variance': 0.0946365497394398, 'contrast': 0.0723099626207696, 'Hue_Std': 0.0704141234636421, 'Value_ThirdMoment': 0.0703988797304429, 'Hue_ThirdMoment': 0.0700739386147031, 'Saturation_Mean': 0.0602725736463129, 'Value_Std': 0.0571275671669869, 'sum_variance': 0.0529912336032915, 'energy': 0.05132550198043, 'variance': 0.050437511074623, 'Saturation_ThirdMoment': 0.0451401440697573, 'max_correlation': 0.0435528984330299, 'Saturation_Std': 0.0357678355234219, 'total_entropy': 0.0349781177176291, 'ASM': 0.0209759478426124}

positive_region = [feature for feature, importance in feature_importances.items() if importance > 0.096]
border_region = [feature for feature, importance in feature_importances.items() if 0.051<= importance <= 0.096]
negative_region = [feature for feature, importance in feature_importances.items() if importance < 0.051]

X_positive = X[positive_region]
svm = SVC(probability=True)
svm.fit(X_positive, y)
acc0 = svm.score(X_positive, y)

def evaluate_model(svm, X, y):
    y_pred = svm.predict(X)
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    return acc, recall, precision, f1

for feature in sorted(border_region, key=feature_importances.get, reverse=True):
    X_positive[feature] = X[feature]
    svm.fit(X_positive, y)
    acc, recall, precision, f1 = evaluate_model(svm, X_positive, y)

    if acc > acc0:
        acc0 = acc
        positive_region.append(feature)
    else:
        X_positive = X_positive.drop([feature], axis=1)

print("Selected Features:", positive_region)
print("Accuracy:", acc0)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)