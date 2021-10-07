# Imports

from bokeh.io import output_notebook, export_png, export_svg
from bokeh.plotting import figure, output_file, show
# from google.colab import drive
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle

# output_notebook()

RANDOM_STATE = 42


# Data loading


# drive.mount("/content/drive")
df = pd.read_csv("results_all.csv")
data, test_data = df[df["test_sample"] == 0], df[df["test_sample"] != 0]
data.pop("test_sample")
test_data.pop("test_sample")


# Data preprocessing


def preprocess_data(df):
  # Remove labels.
  file_names = df.pop("file_name")
  y = [1 if "Virus" in f else 0 for f in file_names]
  if "label" in df.keys():
    y = df.pop("label")
  
  # Label distribution.
  n_negative, n_positive = np.histogram(y, bins=2)[0]
  print(f"Label distribution: positive={n_positive}, negative={n_negative}")

  print("Shape:", df.shape)
  return df, y, file_names

def remove_file_features(df):
  idxes = []
  for i, k in enumerate(df.keys()):
    if "/.wine/" in k or "/wine/" in k or "/wine-" in k or "-wine-" in k:
      continue
    if k.startswith("deleted_") or k.startswith("writte_") or k.startswith("read_") or k.startswith("perms_"):
      continue
    if k[0] in [str(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]:
      continue
    print(k)
    idxes.append(i)
  return df.iloc[:, idxes]

data = remove_file_features(data)
test_data = remove_file_features(test_data)

X, y, _ = shuffle(*preprocess_data(data), random_state=RANDOM_STATE)
X_test, y_test, test_filenames = preprocess_data(test_data)


# Model training


SCORING = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

threshold = .1 * (1 - .1)
print(f"Variance threshold: {threshold}")

# Choose a classifier below:

# coefs:

fi, clf = lambda c: c[-1].coef_.flatten(), make_pipeline(VarianceThreshold(threshold=threshold), StandardScaler(), LogisticRegression())
# fi, clf = lambda c: c[-1].coef_.flatten(), make_pipeline(VarianceThreshold(threshold=threshold), StandardScaler(), LinearSVC(C=1.0))
# fi, clf = lambda c: c[-1].feature_importances_.flatten(), make_pipeline(VarianceThreshold(threshold=threshold), RandomForestClassifier())

# no coefs:

# fi, clf = lambda c: [], make_pipeline(VarianceThreshold(threshold=threshold), StandardScaler(),  SVC(kernel='rbf', C=1.0))
# fi, clf = lambda c: [], make_pipeline(VarianceThreshold(threshold=threshold), StandardScaler(), KNeighborsClassifier(n_neighbors=3))
# fi, clf = lambda c: [], DummyClassifier(strategy="most_frequent")

scores = cross_validate(clf, X, y, scoring=SCORING, cv=5)
for metric_name in [f"test_{score}" for score in SCORING]:
  metric = scores[metric_name]
  metric_arr = "  ".join(f"{x*100.0:.02f}%" for x in metric)
  print(f"{metric_name.split('_')[1] + ':': <10}\t{np.mean(metric)*100.0:.02f}% +- {np.std(metric)*100.0:.02f}%\t\t[{metric_arr}]")

# Re-train on the whole dataset.
clf.fit(X, y)
print("Done.")


# Feature importance


# Features sorted in importance.
top_n = 50
importance = []
for i, (idx, coef) in enumerate(sorted(enumerate(np.abs(fi(clf))), key=lambda x: -x[1])):
  if i < top_n:
    print(f"{str(i+1) + '.': <3} {data.keys()[idx]: <20} (feature: {idx: <4}):\t{coef: <5}")
  importance.append((idx, data.keys()[idx], coef))

# Plot:
features = [x[1] for x in importance]
coef = [x[2] for x in importance]
p = figure(x_range=features[:top_n], plot_height=500, plot_width=600, title=f"Feature importance (top {top_n})",
           tools="save")

p.vbar(x=features, top=coef, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.major_label_orientation = "vertical"

show(p)

# Features sorted in importance.
importance = []
for i, (idx, coef) in enumerate(sorted(enumerate(np.abs(fi(clf))), key=lambda x: -x[1])):
  importance.append((idx, data.keys()[idx], coef))

# Plot:
features = [x[1] for x in importance]
coef = [x[2] for x in importance]
p = figure(x_range=features, plot_height=500, plot_width=1200, title=f"Feature importance (all)",
           tools="save")

p.vbar(x=features, top=coef, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.major_label_orientation = "vertical"

show(p)


# Test set predictions


y_pred_test = clf.predict(X_test)
print(classification_report(y_test, y_pred_test, zero_division=0))
for file_name, y_pred_i, y_test_i in zip(test_filenames, y_pred_test, y_test):
  print(f"{file_name: <50}: predicted: {y_pred_i} actual: {y_test_i}")
  
variance_threshold = VarianceThreshold(threshold=threshold)
variance_threshold.fit(X)


# Feature exploration.


print(f"Shape: {X.shape}")
index = 0
for feature_name, variance in sorted(zip(X.keys(), variance_threshold.variances_), key=lambda x: x[1], reverse=True):
  if variance <= threshold:
    break
  print(f"{index}\t{variance: <16}\t{feature_name}")
  index += 1

print(f"Remaining features: {X.shape[1]-index+1}")


# Variance per groups of features


variance_threshold = VarianceThreshold(threshold=-1)
variance_threshold.fit(X)
print(f"Shape: {X.shape}")
variance_from_feature_name = dict(zip(X.keys(), variance_threshold.variances_))

# print(variance_from_feature_name)

from collections import defaultdict
def get_avg_var(f):
  ips = defaultdict(lambda: 0)
  for i, (k, v) in enumerate(variance_from_feature_name.items()):
    if f(k):
      ips[i] += v
  return ips
ips = get_avg_var(lambda k: k[0].isdigit())
# print(f"IPs:", sum(ips.values()) / len(ips), len(ips))

ips2 = get_avg_var(lambda k: k.startswith("read_"))
# print("Reads:", sum(ips2.values()) / len(ips2), len(ips2))

ips3 = get_avg_var(lambda k: k.startswith("writte_"))
# print("Writes:", sum(ips3.values()) / len(ips3), len(ips3))

ips4 = get_avg_var(lambda k: k.startswith("deleted_"))
# print("Deletes:", sum(ips4.values()) / len(ips4), len(ips4))

ips5 = get_avg_var(lambda k: k.startswith("perms_"))
# print("Perms:", sum(ips5.values()) / len(ips5), len(ips5))

orig = set(i for i, k in enumerate(variance_from_feature_name))
rem = orig-ips.keys()-ips2.keys()-ips3.keys()-ips4.keys()-ips5.keys()-set([166, 167])

ips6keys = set([list(variance_from_feature_name.keys())[i] for i in rem])
ips6 = get_avg_var(lambda k: k in ips6keys)
print("Syscalls:", sum(ips6.values()) / len(ips6), len(ips6))



# Visualization - PCA

n_components = 25
pca = PCA(n_components=n_components)
pca_clf = make_pipeline(VarianceThreshold(threshold), StandardScaler())
transform = pca_clf.fit_transform(X)
transform_test = pca_clf.transform(X_test)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=transform[:,0],
    ys=transform[:,1], 
    zs=transform[:,2], 
    c=['r' if i == 1 else 'b' for i in y], 
)
# ax.scatter(
#     xs=transform_test[:,0],
#     ys=transform_test[:,1], 
#     zs=transform_test[:,2], 
#     c=['k' if i == 1 else 'g' for i in y_test], 
# )
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

R = [255, 0, 0]
B = [0, 0, 255]
G = [0, 255, 0]
K = [0, 0, 0]
colors = np.array([R if label else B for label in y], dtype="uint8")
colors_test = np.array([K if label else G for label in y_test], dtype="uint8")

TOOLS="save"

n_vis = min(3, n_components)
for i in range(n_vis):
  for j in range(i, n_vis):
    if i >= j:
      continue
    p = figure(title="PCA {} vs {}".format(i, j), tools=TOOLS)
    x1 = transform[:, i]
    x2 = transform[:, j]
    p.scatter(x1, x2, fill_color=colors, fill_alpha=1.0,
            line_color=None)
    # x1 = transform_test[:, i]
    # x2 = transform_test[:, j]
    # p.scatter(x1, x2, fill_color=colors_test, fill_alpha=1.0,
    #         line_color=None)
    show(p)
    
    
# Visualization - TSNE

n_components = 3
n_iter = 5_000
verbose = 1
# X_embedded = TSNE(n_components=n_components, verbose=verbose, perplexity=5, n_iter=n_iter, random_state=RANDOM_STATE).fit_transform(VarianceThreshold(threshold).fit_transform(StandardScaler().fit_transform(X)))
# X_embedded = TSNE(n_components=n_components, verbose=verbose, perplexity=50, n_iter=n_iter, random_state=RANDOM_STATE).fit_transform(VarianceThreshold(threshold).fit_transform(StandardScaler().fit_transform(X)))
# X_embedded = TSNE(n_components=n_components, verbose=verbose, perplexity=5, n_iter=n_iter, random_state=RANDOM_STATE).fit_transform(transform)
X_embedded = TSNE(n_components=n_components, verbose=verbose, perplexity=50, n_iter=n_iter, random_state=RANDOM_STATE).fit_transform(transform)

print(X_embedded.shape, y.shape)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X_embedded[:,0],
    ys=X_embedded[:,1], 
    zs=X_embedded[:,2], 
    c=['r' if i == 1 else 'b' for i in y], 
)
ax.set_xlabel('tsne-one')
ax.set_ylabel('tsne-two')
ax.set_zlabel('tsne-three')
plt.show()

for i in range(n_components):
  for j in range(i, n_components):
    if i >= j:
      continue
    p = figure(title="TSNE {} vs {}".format(i, j), tools=TOOLS)
    x1 = X_embedded[:, i]
    x2 = X_embedded[:, j]
    p.scatter(x1, x2, fill_color=colors, fill_alpha=1.0, line_color=None)
    show(p)
