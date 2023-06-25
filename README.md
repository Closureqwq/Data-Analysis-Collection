# Data-Analysis-Collection
"Isolation Forest.py" is an anomaly detection system, specifically designed to uncover latent outliers or anomalous behavior from a dataset. The entire workflow is systematically divided into three critical stages: data preprocessing, model training, and result visualization.

During the data preprocessing phase, the code reads a CSV file and subsequently executes a Discrete Wavelet Transform (DWT) for data transformation, thereby creating two novel series named "cA" and "cD". These series respectively represent the approximation and detail components of the original data.

The next phase is model training. Here, the code employs the Isolation Forest model for training with the primary objective of identifying outliers within the data. In addition, the code utilizes grid search for hyperparameter optimization, with an aim to discover the most optimal parameter configuration.

In the final stage of result visualization, the code predicts the anomaly scores within the test set, and further converts these scores into binary labels. Theoretically, any continuous numerical data with frequency components can substitute the existing dataset.
![image](https://github.com/Closureqwq/Data-Analysis-Collection/assets/102975605/19e2073e-8b06-4e5e-8a3f-426a520751f4)
