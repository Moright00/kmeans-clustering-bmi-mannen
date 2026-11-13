# K-Means Clustering: Gewichtsklasses van Amerikaanse Mannen

## Projectbeschrijving
Ik heb in Python een eigen implementatie van het K-Means clustering algoritme geschreven. Het programma analyseert de dataset `bmi_men.csv` met lengte en gewicht van Amerikaanse mannen en verdeelt deze automatisch in 4 gewichtsklasses (van normaal gewicht tot obesitas klasse 3).

## Hoe het werkt
In plaats van een kant-en-klare machine learning library te gebruiken, is de wiskundige logica zelf opgebouwd:
1. **Start_centroids:** Kiest 4 willekeurige startpunten (*centroïden*).
2. **Distance & centroid_assignation:** Berekent voor elke persoon de afstand tot de centroïden en wijst deze toe aan het dichtstbijzijnde cluster.
3. **K_means:** Herhaalt het toewijzen en herberekenen van het clustergemiddelde totdat de verschuiving kleiner is dan de tolerantie (`1e-8`).
4. **Visualisatie:** Plot alle datapunten in 4 kleuren met vierkante markeringen voor de uiteindelijke centroïden.

## Gebruikte Libraries
* **NumPy:** Voor wiskundige berekeningen en arrays.
* **Pandas:** Voor het inlezen en groeperen van de CSV-dataset.
* **Matplotlib:** Voor het visualiseren van de clusters in een spreidingsdiagram.
