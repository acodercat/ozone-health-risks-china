# Machine Learning-Driven Spatiotemporal Analysis of Ozone Exposure and Health Risks in China

This repository contains the code and datasets for the research project "Machine Learning-Driven Spatiotemporal Analysis of Ozone Exposure and Health Risks in China". The project aims to analyze the spatiotemporal patterns of ozone exposure and associated health risks in China using machine learning techniques.

## Dataset

Due to the large size of the dataset files, they are hosted on Google Drive. Please download the following files from the provided Google Drive link and place them in the `dataset` directory:

- Google Drive: [https://drive.google.com/drive/folders/1y98XDdTym5saE1U98IlvXoBnAwabIIYo](https://drive.google.com/drive/folders/1y98XDdTym5saE1U98IlvXoBnAwabIIYo)

Files to download:
- `grid_features.csv`: Features for each grid cell.
- `grid_static_features.csv`: Static features for each grid cell.
- `test_grid_ids.csv`: IDs of the grid cells used for testing the machine learning models.
- `train_grid_ids.csv`: IDs of the grid cells used for training the machine learning models.
- `trainset_daily.csv`: Training dataset with daily temporal resolution.
- `trainset_monthly.csv`: Training dataset with monthly temporal resolution.
- `trainset_weekly.csv`: Training dataset with weekly temporal resolution.

### Train and Test Grid IDs

The files `train_grid_ids.csv` and `test_grid_ids.csv` play a crucial role in the project. These files contain the IDs of the grid cells that are used for training and testing the machine learning models, respectively.

- `TRAIN_GRID_IDS_PATH = './dataset/train_grid_ids.csv'`: This variable stores the path to the `train_grid_ids.csv` file, which contains the IDs of the grid cells used for training the models. The machine learning algorithms will learn from the data associated with these grid cells to build the predictive models.

- `TEST_GRID_IDS_PATH = './dataset/test_grid_ids.csv'`: This variable stores the path to the `test_grid_ids.csv` file, which contains the IDs of the grid cells used for testing the models. The trained models will be evaluated on the data associated with these grid cells to assess their performance and generalization ability.

By separating the grid cells into training and testing sets, we ensure that the models are evaluated on unseen data, providing a fair assessment of their predictive capabilities.

## NetGBM: Neural Network Gradient Boosting Machine

The `NetGBM.ipynb` notebook contains the implementation of the main algorithm structure used in this research paper. NetGBM, short for Neural Network Gradient Boosting Machine, is a novel approach that combines the strengths of neural networks and gradient boosting machines for spatiotemporal modeling of ozone exposure and health risks.

Key features of NetGBM:

1. **Neural Network Base Learner**: NetGBM uses a neural network as the base learner in the gradient boosting framework. The neural network captures complex nonlinear relationships between the input features and the target variable (ozone concentration).

2. **Gradient Boosting**: The gradient boosting algorithm iteratively trains a series of neural network base learners, each focusing on the residuals (errors) of the previous learners. This allows NetGBM to progressively refine the predictions and capture intricate patterns in the data.

3. **Spatiotemporal Modeling**: NetGBM incorporates spatial and temporal information by utilizing features that represent the geographical location and time-varying characteristics of each grid cell. This enables the model to capture the spatiotemporal dynamics of ozone exposure and health risks.

The `NetGBM.ipynb` notebook provides a step-by-step implementation of the NetGBM algorithm, including data preprocessing, model training, and evaluation. It also includes visualizations and analysis of the model's performance and interpretability.

For more details on the NetGBM algorithm and its application in this research, please refer to the `NetGBM.ipynb` notebook and the accompanying research paper.

## Code

The repository includes various Python scripts and Jupyter notebooks for data preprocessing, model training, evaluation, and visualization. Some of the key files include:

- `lgb.py` and `xgb.py`: LightGBM and XGBoost model implementations.
- `nn.py`: Neural network model implementation.
- `evaluation_metrics.py`: Evaluation metrics calculation.
- `visualization/`: Directory containing scripts for data visualization.


## Usage

To run the code and reproduce the results, follow these steps:

1. Download the required dataset files from the provided Google Drive link and place them in the `dataset` directory.
2. Install the required dependencies listed in `pyproject.toml`.
3. Run the desired scripts or notebooks to preprocess the data, train the models, and generate the results.
4. Use the visualization scripts in the `visualization/` directory to create maps, plots, and other visualizations of the results.

For more details on each script and notebook, please refer to the comments and documentation within the files.

## License

This project is licensed under the [MIT License](LICENSE).