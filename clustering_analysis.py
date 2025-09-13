import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/retail_data.csv')
X = data[['feature1', 'feature2']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X)
sns.scatterplot(x='feature1', y='feature2', hue='cluster', data=data)
plt.show()
