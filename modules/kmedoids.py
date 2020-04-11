from pyclustering.cluster.kmedoids import kmedoids as kmedoids_
from pyclustering.utils import distance_metric, type_metric
import numpy as np

def kmedoids(data):
    x = np.array(data["x"], dtype='float')
    y = np.array(data["y"], dtype='float')

    if len(x) == 0:
        return {
            'centroids': [],
            'points': []
        }

    k = int(data['k'])

    D = np.array([x, y]).T
    k = min(k, D.shape[0])
    
    metrics = {
        "manhattan": distance_metric(type_metric.MANHATTAN, data=D),
        "euclidean": distance_metric(type_metric.EUCLIDEAN, data=D),
        "chebyshev": distance_metric(type_metric.CHEBYSHEV, data=D),
        "canberra": distance_metric(type_metric.CANBERRA, data=D),
        "chi-square": distance_metric(type_metric.CHI_SQUARE, data=D) 
    }
    
    metric = metrics[data['metric']]
    kmedoids_instance = kmedoids_(D, list(range(k)), metric=metric)
    kmedoids_instance.process()

    labels = kmedoids_instance.predict(D)
    medoids = np.array(kmedoids_instance.get_medoids())
    medoids = D[medoids]

    output_data = {
        'centroids': [{'x': medoids[i, 0], 'y': medoids[i, 1], 'label': i} for i in range(len(medoids))],
        'points': [{'x': D[i, 0], 'y': D[i, 1], 'label': int(labels[i])} for i in range(len(labels))]
    }

    return output_data
