import os
import random

from typing import List
from pathlib import Path

import numpy as np

from tqdm import tqdm
from bonapity import bonapity

from sklearn.svm import OneClassSVM, LinearSVC, SVC
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


random.seed(0xBED4BABE)
np.random.seed(0xBED4BABE)

images_path   = list(Path("images/").glob("*.jpg"))
features_path = list(Path("features/").glob("*.feat"))

images_ids   = set([p.with_suffix('').name for p in images_path])
features_ids = set([p.with_suffix('').name for p in features_path])

to_keep = features_ids & images_ids

print(f"Exceeding images : {  ','.join(sorted(  images_ids - to_keep)) if len(to_keep) != len(images_ids) else 'None'}")
print(f"Exceeding features : {','.join(sorted(features_ids - to_keep)) if len(to_keep) != len(features_ids) else 'None'}")

to_keep = sorted(to_keep)

images_path   = sorted([p for p in   images_path if p.with_suffix('').name in to_keep])
features_path = sorted([p for p in features_path if p.with_suffix('').name in to_keep])

features = np.array([np.load(str(p), allow_pickle=True) for p in tqdm(features_path)])


# First heuristic trying to find few anomalies to label as negative
anomaly_detector = IsolationForest()
anomaly_detector.fit(features)

# Learn a model reproducing the anomaly to predict a probability score later
model = RandomForestClassifier(class_weight="balanced")
model.fit(features, anomaly_detector.predict(features))

soft_labels = model.predict_proba(features) # Probabilities
model_labels = soft_labels[:, 0] < .5
user_labels = np.zeros(len(features)) # +1 belong to the class, -1 do not belong to the class

# Start of the correction loop
@bonapity
def get_model_labels() -> List[int]:
    return list(model_labels.astype(int) * 2 - 1)

@bonapity
def get_model_confidence() -> List[float]:
    return list(soft_labels.T[0])#1 - abs(soft_labels.T[0] - .5).astype(float)) #FIXME:?

@bonapity
def get_user_labels() -> List[int]:
    return list(user_labels.astype(int))

@bonapity
def get_ids() -> List:
    return to_keep

@bonapity
def set_label(image_id: str, label: int):
    global user_labels, to_keep
    if not label in [-1, 1]:
        return "`label` should be in {-1, 1}…"
    user_labels[to_keep.index(image_id)] = label
    return True # Return that the change have been saved

@bonapity
def refine_learning() -> List[float]:
    global model, soft_labels, model_labels
    if not (-1 in list(user_labels.astype(int)) and 1 in list(user_labels.astype(int))):
        return "you should label at least 2 images (one positive and one negative)…"
    model = LogisticRegression(class_weight='balanced', l1_ratio=.5) #RandomForestClassifier(n_estimators=1, class_weight="balanced") #SVC(C=.1, class_weight="balanced", probability=True)
    model.fit(features, user_labels)
    soft_labels = model.predict_proba(features)
    model_labels = soft_labels[:, 0] < .5
    return  model_labels# Probabitity of being positive

@bonapity
def dowload_list_to_keep() -> List[str]:
    global model
    return '\n'.join([str(p) for p, k in zip(images_path, [(user if user != 0 else pred) for pred, user in zip(model.predict(features), user_labels)]) if k])

@bonapity(mime_type="auto")
def get_image(image_id: str):
    global to_keep, images_path
    return open(images_path[to_keep.index(image_id)], 'rb').read()


if __name__ == "__main__":
    bonapity.serve()



