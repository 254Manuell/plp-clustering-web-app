import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def can_be_clustered(df):
    # Check if there are at least 2 numeric columns and at least 5 rows
    numeric = df.select_dtypes(include=['number'])
    return numeric.shape[1] >= 2 and numeric.shape[0] >= 5


def perform_clustering(df, n_clusters=3):
    numeric = df.select_dtypes(include=['number'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return numeric, labels


def save_plot(numeric, labels, plot_path):
    if numeric.shape[1] == 2:
        plt.figure(figsize=(8,6))
        plt.scatter(numeric.iloc[:,0], numeric.iloc[:,1], c=labels, cmap='viridis')
        plt.xlabel(numeric.columns[0])
        plt.ylabel(numeric.columns[1])
        plt.title('KMeans Clustering')
        plt.savefig(plot_path)
        plt.close()
    else:
        # Use pairplot for more than 2 dimensions
        numeric['Cluster'] = labels
        sns.pairplot(numeric, hue='Cluster', palette='viridis')
        plt.savefig(plot_path)
        plt.close()
        numeric.drop('Cluster', axis=1, inplace=True)

from sklearn.metrics import silhouette_score as skl_silhouette_score

@app.route('/', methods=['GET', 'POST'])
def index():
    columns = []
    selected_features = []
    k = 3
    standardize = False
    plot_url = None
    silhouette = None
    cluster_summary = None
    if request.method == 'POST':
        # Get parameters from form
        k = int(request.form.get('k', 3))
        standardize = 'standardize' in request.form
        selected_features = request.form.getlist('features')
        # File upload
        file = request.files.get('file')
        if file and file.filename != '':
            try:
                df = pd.read_csv(file)
            except Exception as e:
                flash(f'Could not read file: {e}')
                return render_template('index.html', columns=columns, selected_features=selected_features, k=k, standardize=standardize)
            numeric = df.select_dtypes(include=['number'])
            columns = list(numeric.columns)
            # If no features selected, default to all
            if not selected_features or not set(selected_features).issubset(set(columns)):
                selected_features = columns
            if len(selected_features) < 2 or numeric.shape[0] < 5:
                flash('Select at least 2 numeric features and ensure at least 5 rows.')
                return render_template('index.html', columns=columns, selected_features=selected_features, k=k, standardize=standardize)
            X = numeric[selected_features]
            if standardize:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
            except Exception as e:
                flash(f'Clustering error: {e}')
                return render_template('index.html', columns=columns, selected_features=selected_features, k=k, standardize=standardize)
            # Silhouette score
            try:
                silhouette = skl_silhouette_score(X_scaled, labels)
                silhouette_disp = f"{silhouette:.2f}"
            except Exception:
                silhouette_disp = '--'
            # Cluster summary
            cluster_summary = []
            try:
                for cl in range(k):
                    mask = (labels == cl)
                    size = mask.sum()
                    # Per-cluster silhouette
                    if size > 1:
                        sil = skl_silhouette_score(X_scaled[mask], [cl]*size)
                    else:
                        sil = float('nan')
                    cluster_summary.append({'cluster': f'Cluster {cl+1}', 'size': size, 'silhouette': f"{sil:.2f}" if not pd.isna(sil) else '--'})
            except Exception:
                cluster_summary = None
            # Save plot
            plot_filename = f"cluster_{uuid.uuid4().hex}.png"
            plot_path = os.path.join('static', plot_filename)
            # Visualization
            if len(selected_features) == 2:
                plt.figure(figsize=(8,6))
                plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='tab10')
                plt.xlabel(selected_features[0])
                plt.ylabel(selected_features[1])
                plt.title('KMeans Clustering')
                plt.savefig(plot_path)
                plt.close()
            else:
                plot_df = X.copy()
                plot_df['Cluster'] = labels
                sns.pairplot(plot_df, hue='Cluster', palette='tab10')
                plt.savefig(plot_path)
                plt.close()
            plot_url = url_for('static', filename=plot_filename)
            return render_template('index.html', plot_url=plot_url, columns=columns, selected_features=selected_features, k=k, standardize=standardize, silhouette_score=silhouette_disp, cluster_summary=cluster_summary)
        else:
            flash('Please upload a CSV file.')
            return render_template('index.html', columns=columns, selected_features=selected_features, k=k, standardize=standardize)
    # GET
    return render_template('index.html', columns=columns, selected_features=selected_features, k=k, standardize=standardize, silhouette_score='--', cluster_summary=None)

if __name__ == '__main__':
    app.run(debug=True)
