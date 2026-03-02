import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster


def nan_replace(tabel: pd.DataFrame) -> None:
    """Completarare valori lipsa (numeric=medie, text=mode)."""
    for col in tabel.columns:
        if tabel[col].isna().any():
            if is_numeric_dtype(tabel[col]):
                tabel[col].fillna(tabel[col].mean(), inplace=True)
            else:
                tabel[col].fillna(tabel[col].mode()[0], inplace=True)


def partitie(h, nr_clusteri: int, p: int, instante: list[str]) -> np.ndarray:
    """Construire partitie pentru un numar fix de clustere (taiere dendrogramă)."""
    # Prag de tăiere (cut-off) pentru dendrograma
    k_diff = p - nr_clusteri
    prag = (h[k_diff, 2] + h[k_diff + 1, 2]) / 2

    # Dendrograma pentru nr_clusteri
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Clusterizare ierarhica (Ward) – {nr_clusteri} clustere")
    hclust.dendrogram(h, labels=instante, ax=ax, color_threshold=prag)
    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()

    # Inițializare etichete (fiecare instanta în cluster separat)
    n = p + 1
    c = np.arange(n)

    # Simulare reuniuni pana raman nr_clusteri
    for i in range(n - nr_clusteri):
        k1 = int(h[i, 0])
        k2 = int(h[i, 1])
        c[c == k1] = n + i
        c[c == k2] = n + i

    # Codificare finală clustere
    coduri = pd.Categorical(c).codes
    return np.array([f"C{cod + 1}" for cod in coduri])


def histograma(x: np.ndarray, variabila: str, partitia: np.ndarray) -> None:
    """Histograma pe clustere pentru o variabilă."""
    clustere = np.unique(partitia)
    fig, axs = plt.subplots(1, len(clustere), figsize=(4 * len(clustere), 4), sharey=True)
    if len(clustere) == 1:
        axs = [axs]

    fig.suptitle(f"Distribuție pe clustere: {variabila}")

    for ax, cluster in zip(axs, clustere):
        ax.hist(x[partitia == cluster], bins=10, rwidth=0.9)
        ax.set_title(cluster)

    plt.tight_layout()


def execute():
    # 1) Citire date
    df = pd.read_csv("world-happiness-report-2021.csv")

    # 2) Setare index = țară
    df = df.set_index("Country name")
    instante = list(df.index)

    # 3) Selectare variabile pt clusterizare
    variabile = [
        "Logged GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption",
    ]

    # 4) Curățare NA + matrice X
    nan_replace(df)
    X = df[variabile].values

    # 5) Standardizare variabile (z-score)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 6) Clusterizare ierarhica (Ward) -> matrice linkage
    h = hclust.linkage(X_std, method="ward")
    n = len(instante)
    p = n - 1

    # 7) Partiții pentru k=2..5 + export CSV
    for k in [2, 3, 4, 5]:
        part_k = partitie(h, k, p, instante)
        out = pd.DataFrame({"Cluster": part_k}, index=instante)
        out.to_csv(f"Partitie_{k}_clusteri_happiness_2021.csv")

    # 8) Estimare k optim
    k_diff_max = np.argmax(h[1:, 2] - h[:-1, 2])
    k_opt = p - k_diff_max
    print(f"Numar 'optim' de clustere (heuristic): {k_opt}")

    # 9) Partiția finală (k optim) + atasare în tabel
    part_opt = partitie(h, k_opt, p, instante)
    df["Cluster"] = part_opt

    # 10) Verificare alternativă (fcluster)
    auto = fcluster(h, k_opt, criterion="maxclust")
    print(f"fcluster (maxclust={k_opt}) labels (primele 10): {auto[:10]}")

    # 11) Profil clustere (medii pe variabile)
    profile = df.groupby("Cluster")[variabile].mean().round(3)
    print("\nMediile variabilelor pe cluster:")
    pd.set_option("display.max_columns", None)
    print(profile)

    # 12) Medie Ladder score pe cluster (interpretare)
    if "Ladder score" in df.columns:
        ladder_means = df.groupby("Cluster")["Ladder score"].mean().round(3)
        print("\nMedia Ladder score pe cluster:")
        print(ladder_means)

    # 13) Regiune x cluster (tabel de frecvențe)
    if "Regional indicator" in df.columns:
        region_table = pd.crosstab(df["Regional indicator"], df["Cluster"])
        print("\nTabel regiune x cluster:")
        print(region_table)

    # 14) Grafice rapide (histograme pe primele 3 variabile)
    for i in range(min(3, len(variabile))):
        histograma(X[:, i], variabile[i], part_opt)

    # 15) Afișare grafice
    plt.show()


if __name__ == "__main__":
    execute()
