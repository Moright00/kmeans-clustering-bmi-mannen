# ==============================================================================
# Project: K-Means Clustering op BMI-dataset van Amerikaanse Mannen
# Omgeving: Spyder (Python)
# Beschrijving: Eigen implementatie van het K-Means algoritme om lengte (cm) en 
#               gewicht (kg) te groeperen in 4 gewichtsklasses.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Voor het gemak werd dit project op spyder gemaakt

bmi = pd.read_csv('bmi_men.csv')
colors = ListedColormap(["red", "purple", "green", "blue"])
np.random.seed(6)    # Zorgt voor reproduceerbare willekeurige startpunten
k = 4                # Aantal clusters (gewichtsklasses)


def Start_centroids(k, dataset):
    """Kiest k willekeurige datapunten als startpunten (centroïden)."""
    centroids = dataset.sample(k)
    return centroids


def Distance(x, y):
    """Berekent de afstand tussen twee punten."""
    d = np.square(np.sum((x - y) ** 2))
    return d


def centroid_assignation(dataset, centroids):
    """Wijst elke observatie toe aan de dichtstbijzijnde centroïde."""
    centroid_assign = []
    error_assign = []
    k = centroids.shape[0]
    n = dataset.shape[0]

    for obs in range(n):
        full_errors = np.array([])
        for centroid in range(k):
            # Bereken de afstand (error) tussen huidige centroïde en observatie
            err = Distance(centroids.iloc[centroid, :], dataset.iloc[obs, :])  # error tussen actuele centroid en actuele observatie
            full_errors = np.append(full_errors, err)

        # Zoek de dichtstbijzijnde centroïde en de bijbehorende kleinste error
        closest_centroid = np.where(full_errors == np.amin(full_errors))[0].tolist()[0]  # centroid met minst error
        closest_centroid_error = np.amin(full_errors)                                    # waarde min error bijhouden

        centroid_assign.append(closest_centroid)
        error_assign.append(closest_centroid_error)
    return centroid_assign, error_assign


def K_means(dataset, k, tol=1e-8):
    """Voert het K-Means clustering algoritme iteratief uit."""
    copy_dataset = dataset.copy()
    err = []
    start = True
    x = 0

    centroids = Start_centroids(k, dataset)

    while(start):
        copy_dataset['centroid'], x_err = centroid_assignation(copy_dataset, centroids)
        err.append(sum(x_err))
        
        # Herbereken de positie van de centroïden op basis van het gemiddelde per cluster
        centroids = copy_dataset.groupby('centroid').agg('mean').reset_index(drop=True)
        
        # Stop de loop zodra de verbetering kleiner is dan de tolerantie (tol)
        if x > 0:
            if err[x - 1] - err[x] <= tol:
                start = False
        x += 1

    copy_dataset['centroid'], x_err = centroid_assignation(copy_dataset, centroids)
    centroids = copy_dataset.groupby('centroid').agg('mean').reset_index(drop=True)
    return copy_dataset['centroid'], x_err, centroids

# --- Uitvoering van het algoritme ---
bmi_coor = bmi[['Height', 'Weight']]
centroids = Start_centroids(k, bmi_coor)

bmi_coor['centroid'], bmi_coor['error'], centroids = K_means(bmi_coor[['Height', 'Weight']], 4)
bmi_coor.head()

# --- Visualisatie van de clusters ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(bmi_coor.iloc[:, 0], bmi_coor.iloc[:, 1], c=bmi_coor['centroid'].astype('category'), cmap=colors, s=100, alpha=0.6)
plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], marker='s', s=200, c=[0, 1, 2, 3], cmap=colors)
ax.set_title('K-Means Clustering: Lengte vs. Gewicht (Amerikaanse Mannen)')
ax.set_xlabel('cm')
ax.set_ylabel('kg')
plt.show()
