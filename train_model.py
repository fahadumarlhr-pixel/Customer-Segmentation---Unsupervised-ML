import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Loadind the Dataset :- 
df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# Remove duplicates :- 
df = df.drop_duplicates()

# Rename columns to have easy names and avoid complex :- 
df = df.rename(columns={
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score"
})


# Select Features (Input Features) :-
features = [
    "Age",
    "Annual_Income",
    "Spending_Score"
]

X = df[features]

# Applying Standard Scaler :-
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method :-
inertia = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10       # 10 Centroids
    )

    model.fit(X_scaled)
    inertia.append(model.inertia_)

# Plotting the Results :-
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid()
plt.savefig("elbow_method.png")
plt.close()

# ----------------------------------------------------------
# SILHOUETTE SCORE :-
silhouette_scores = {}

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10       # 10 Centroids 
    )

    labels = model.fit_predict(X_scaled)  # Divide Customers into Clusters

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


# Select best K :- 
best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

print(
    f"\nBest number of clusters: {best_k}"
)

# ---------------------------------------------------
# Final K-means model :-
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


print("\n========== CLUSTER-COUNTS ==========\n")
print(df["Cluster"].value_counts().sort_index())

# CLUSTER ANALYSIS :-
cluster_analysis = df.groupby("Cluster")[features].mean()

print("\n============ CLUSTER ANALYSIS ============\n")
print(cluster_analysis)
 
# PCA :-   (Principle Component Analysis)
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA_1"] = X_pca[:, 0]
df["PCA_2"] = X_pca[:, 1]

# PCA's Result Visualization :- 
plt.figure(figsize=(9, 6))

plt.scatter(
    df["PCA_1"],
    df["PCA_2"],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments")
plt.colorbar(label="Cluster")
plt.grid()
plt.savefig(
    "customer_clusters.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Saving all the Results :-
# -> Converting the Results into CSV file  

df.to_csv(
    "segmented_customers.csv",
    index=False
)

# -> Saving the Model for creating the app :-
joblib.dump(
    kmeans,
    "kmeans_model.pkl"
)

# -> Save Scaled Content :-
joblib.dump(
    scaler,
    "scaler.pkl"
)


# -> Save feature names :-
joblib.dump(
    features,
    "features.pkl"
)


print("\n\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==================================")
