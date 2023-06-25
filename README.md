# Data-Analysis-Collection
"Isolation Forest.py" constitutes a sophisticated anomaly detection system, meticulously engineered to uncover latent outliers or anomalous behavior within a dataset. The entire workflow is methodically segmented into three key stages, namely: data preprocessing, model training, and result visualization.

In the data preprocessing stage, the code peruses a CSV file, subsequently executing a Discrete Wavelet Transform (DWT) for data transformation, thereby generating two novel series dubbed "cA" and "cD". These series are emblematic of the approximate and detail components of the original data, respectively.

Following this, the model training phase comes into play. Here, the code employs the Isolation Forest model for training, with the primary aim of identifying outliers within the data. Additionally, grid search is deployed for hyperparameter optimization, geared towards locating the most optimal parameter configuration.

During the final stage of result visualization, the code predicts the anomaly scores within the test set, further transmuting these scores into binary labels. Theoretically, any continuous numerical data with frequency components can be used as a substitute for the existing dataset.
