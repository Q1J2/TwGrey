import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


df = pd.read_excel(r"D:\UCM\UCM100\100features.xlsx")


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


feature_importances = {'correlation': 0.5873189928568828, 'Hue_Mean': 0.5714800057277086, 'sum_entropy': 0.5685793458775091, 'max_correlation': 0.5682303043197512, 'Hue_Std': 0.5678106644352817, 'Hue_ThirdMoment': 0.5673271825465579, 'diff_variance': 0.5671778461273931, 'Value_Mean': 0.5662447186241355, 'total_entropy': 0.5656809997423786, 'mean': 0.5640548374618726, 'sum_mean': 0.5640069351158651, 'Saturation_ThirdMoment': 0.5625655652822533, 'Saturation_Mean': 0.561311020058864, 'dissimilarity': 0.561052674328956, 'Saturation_Std': 0.560698338491803, 'Value_Std': 0.5598652962137907, 'energy': 0.5598511791129184, 'Value_ThirdMoment': 0.55955879700984, 'mutual_information': 0.5576294559734329, 'variance': 0.5563587124079378, 'homogeneity': 0.5550239123126091, 'contrast': 0.5550069929970823, 'ASM': 0.5540108717587878, 'diff_entropy': 0.5433231964511883, 'sum_variance': 0.5349146247477675}


positive_region = [feature for feature, importance in feature_importances.items() if importance > 0.560]
border_region = [feature for feature, importance in feature_importances.items() if 0.545<= importance <= 0.560]
negative_region = [feature for feature, importance in feature_importances.items() if importance < 0.545]


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