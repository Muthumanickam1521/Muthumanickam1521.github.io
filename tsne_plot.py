import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import plotly.express as px

# 1. Create synthetic clustered data
X, y = make_blobs(n_samples=1000, centers=10, cluster_std=1.2, random_state=42)

# 2. Apply t-SNE
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X)

# 3. Create dataframe for plot
df = pd.DataFrame(X_tsne, columns=['x', 'y'])
df['cluster'] = y.astype(str)  # convert to string for color mapping

# 4. Plot using plotly
fig = px.scatter(
    df, x='x', y='y', color='cluster',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title='✨ Beautiful TSNE Clustering - 10 Clusters',
    hover_data=['cluster']
)

# Optional: customize theme
fig.update_layout(
    template='plotly_white',
    title_font_size=24,
    title_x=0.5,
    height=700,
    margin=dict(l=20, r=20, t=60, b=20)
)

# 5. Save as interactive HTML
fig.write_html("tsne_clusters.html")
print("✅ HTML saved as tsne_clusters.html")
