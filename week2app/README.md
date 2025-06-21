# ClusterGenie

![ClusterGenie Screenshot](static/cluster_readme_screenshot.png)

ClusterGenie is a modern, user-friendly Flask web app for performing machine learning clustering on your datasets. Upload your CSV, select features, choose the number of clusters, and visualize the results with interactive, beautiful plots. The app also provides silhouette scores and a cluster summary for deeper insights.

## Features
- Upload your own CSV data
- Select features and number of clusters (K)
- K-Means clustering (more algorithms coming soon)
- Option to standardize features
- Interactive cluster visualization (2D scatter or pairplot)
- Silhouette score and per-cluster summary
- Modern, responsive UI (desktop & mobile)

## Screenshot
![ClusterGenie UI](static/cluster_readme_screenshot.png)

## Getting Started

### Prerequisites
- Python 3.8+

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/254Manuell/plp-clustering-web-app.git
   cd plp-clustering-web-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
1. Click **Upload Data** and select your CSV file.
2. Choose the number of clusters (K) and features to use.
3. (Optional) Toggle standardization.
4. Click **Run Clustering**.
5. View the visualization, silhouette score, and cluster summary.

## Example CSV Format
```
feature1,feature2,feature3
1.2,3.4,5.6
2.3,4.5,6.7
...
```

## GitHub Actions (CI)
This project is ready for CI/CD with GitHub Actions. See `.github/workflows/python-app.yml` for details.

## License
MIT

---

*ClusterGenie – Clustering made easy!*
