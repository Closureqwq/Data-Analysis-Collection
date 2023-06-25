import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pywt
from pywt import dwt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots 

def preprocess_data(df):
    cA, cD = pywt.dwt(df['Close'].values, 'haar')
    df_transformed = pd.DataFrame({"cA": cA, "cD": cD})
    df_transformed.fillna(-99999, inplace=True)
    return df_transformed

def train_model(df):
    print("Training model...")
    X = df
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    param_grid = {'contamination': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300], 'max_samples': ['auto', 0.5], 'max_features': [1, 2]}

    total_combinations = len(param_grid['contamination'])*len(param_grid['n_estimators'])*len(param_grid['max_samples'])*len(param_grid['max_features'])

    best_score = np.inf
    best_params = None

    pbar = tqdm(total=total_combinations)

    for contamination in param_grid['contamination']:
        for n_estimators in param_grid['n_estimators']:
            for max_samples in param_grid['max_samples']:
                for max_features in param_grid['max_features']:
                    model = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=42)
                    model.fit(X_train)
                    score = -model.score_samples(X_train).mean()
                    if score < best_score:
                        best_score = score
                        best_params = {'contamination': contamination, 'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}
                    pbar.update(1)
    
    pbar.close()

    print("Best parameters found: ", best_params)
    print("Best score found: ", best_score)
    
    model = IsolationForest(**best_params, random_state=42)
    model.fit(X_train)
    
    return model, X_test, scaler


def main():
    df = pd.read_csv("Your file name")
    df.index = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.resample('1H').mean()
    df = df.dropna(subset=['Close'])

    df_transformed = preprocess_data(df)
    model, X_test, scaler = train_model(df_transformed)
    
    anomaly_scores_test = -model.score_samples(X_test)
    print("Mean anomaly score (test): ", anomaly_scores_test.mean())
    
    threshold = np.percentile(anomaly_scores_test, 95)
    anomalies = anomaly_scores_test > threshold
    
    print(f"Number of anomalies detected: {anomalies.sum()}")
    print(f"Rate of anomalies: {anomalies.sum() / len(anomalies) * 100:.2f}%")

    X_test_original = scaler.inverse_transform(X_test)

    fig = make_subplots(rows=2, cols=1)

    # Visualizing the data and the anomalies
    anomalies_indices = np.where(anomalies)
    
    anomaly_scores_normalized = (anomaly_scores_test - anomaly_scores_test.min()) / (anomaly_scores_test.max() - anomaly_scores_test.min())
    anomaly_colors = plt.get_cmap('Reds')(anomaly_scores_normalized)

    fig.add_trace(go.Scatter(x=df.index[:len(X_test_original)], y=X_test_original[:, 0], mode='lines', name='cA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[:len(X_test_original)], y=X_test_original[:, 1], mode='lines', name='cD', line=dict(color='green')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index[:len(X_test_original)], y=pd.Series(X_test_original[:, 0]).rolling(window=20).mean(), mode='lines', name='Trend in cA', line=dict(color='violet')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[:len(X_test_original)], y=pd.Series(X_test_original[:, 1]).rolling(window=20).mean(), mode='lines', name='Trend in cD', line=dict(color='violet')), row=2, col=1)

    for i in anomalies_indices[0]:
        fig.add_trace(go.Scatter(x=[df.index[i]], y=[X_test_original[i, 0]], mode='markers', marker=dict(color=np.array([anomaly_colors[i]]), size=6), hovertext=f'Anomaly score: {anomaly_scores_test[i]:.2f}, Index: {df.index[i]}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[df.index[i]], y=[X_test_original[i, 1]], mode='markers', marker=dict(color=np.array([anomaly_colors[i]]), size=6), hovertext=f'Anomaly score: {anomaly_scores_test[i]:.2f}, Index: {df.index[i]}'), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=True, title_text="Time")  # Add a range slider to the x-axis and a label
    fig.update_yaxes(title_text="cA", row=1, col=1)  
    fig.update_yaxes(title_text="cD", row=2, col=1)  
    
    fig.update_layout(height=800, title_text="cA and cD with Anomalies", showlegend=True)

    fig.show()



if __name__ == "__main__":
    main()
