import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import cv2
import os
NUM_CLUS = 50

print("Reading data...")
sift = cv2.SIFT_create()
dir = 'training/'
features = []
classes = set()
for filename in sorted(os.listdir(dir)):
  if filename[0] != '.':
      image = cv2.imread(dir + filename, 0)
      feat, desc = sift.detectAndCompute(image, None)
      features.append([filename, feat, desc])
      classes.add(filename[:3])
print("Categories detected:")
for c in classes:
    print(c)

print("Learning...")
descriptors = np.vstack([des[2] for des in features])
kmeans = KMeans(n_clusters=NUM_CLUS).fit(descriptors)

hists = []
for data in features:
  hists.append((data[0], np.histogram(kmeans.predict(data[2]), bins = NUM_CLUS)[0] / len(data[1])))

print("Classifying...")
sift = cv2.SIFT_create()
dir = 'testing/'
features_test = []
for filename in sorted(os.listdir(dir)):
  if filename[0] != '.':
      image = cv2.imread(dir + filename, 0)
      feat, desc = sift.detectAndCompute(image, None)
      features_test.append([filename, feat, desc])
hists_test = []
for data in features_test:
  hists_test.append((data[0], np.histogram(kmeans.predict(data[2]), bins = NUM_CLUS)[0] / len(data[1])))

model = RandomForestClassifier()
X_train = [x[1] for x in hists]
X_test = [x[1] for x in hists_test]
y = [x[0][0:3] for x in hists]
model.fit(X_train, y)
predictions = model.predict(X_test)
for i in range(0, len(features_test)):
  print(features_test[i][0] + ":\t" + predictions[i])
