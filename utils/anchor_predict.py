from sklearn.cluster import KMeans
import numpy as np


def anchor_prediction(w: list, h: list, n_clusters: int, original_height: int = 640, original_width: int = 640,
                      c_number: int = 416):
    w = np.asarray(w)
    h = np.asarray(h)
    x = [w, h]
    x = np.asarray(x)
    x = np.transpose(x)
    print(x.shape)
    k_mean = KMeans(n_clusters=n_clusters)
    k_mean.fit(x)
    predicted_anchors = k_mean.predict(x)
    anchors = []
    for idx in range(n_clusters):
        anchors.append(np.mean(x[predicted_anchors == idx], axis=0))
        print(f'anchor Added : {np.mean(x[predicted_anchors == idx], axis=0)}')
    anchors = np.array(anchors)
    anchors_copy = anchors.copy()
    anchors[..., 0] = anchors_copy[..., 0] / original_width * c_number
    anchors[..., 1] = anchors_copy[..., 1] / original_height * c_number
    anchors = np.rint(anchors)
    anchors.sort(axis=0)
    return anchors
