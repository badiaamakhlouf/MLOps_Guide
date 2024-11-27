import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HousePricePredictionPipeline:
    def __init__(self):
        """
        Initialize the house price prediction pipeline
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
        self.feature_names = None

    def prepare_data(self, X, y):
        """
        Prepare data for training

        :param X: Feature matrix
        :param y: Target prices
        :return: Prepared feature matrix and target
        """
        # Convert input to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Store feature names
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None

        return X, y

    def train(self, X, y):
        """
        Train the regression pipeline

        :param X: Training features
        :param y: Training target prices
        """
        X, y = self.prepare_data(X, y)
        self.pipeline.fit(X, y)

    def predict(self, X):
        """
        Predict house prices

        :param X: Features to predict
        :return: Predicted prices
        """
        # Ensure input is in the correct format
        if not isinstance(X, pd.DataFrame) and self.feature_names:
            X = pd.DataFrame(X, columns=self.feature_names)

        return self.pipeline.predict(X)

    def get_metrics(self, y_true, y_pred):
        """
        Calculate regression performance metrics

        :param y_true: True house prices
        :param y_pred: Predicted house prices
        :return: Dictionary of performance metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

class TestHousePricePredictionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up test data for the entire test class
        """
        # Simulated house features dataset
        np.random.seed(42)
        n_samples = 200

        # Simulate features
        cls.X = pd.DataFrame({
            'square_feet': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'year_built': np.random.randint(1950, 2023, n_samples),
            'distance_to_city_center': np.random.normal(5, 2, n_samples)
        })

        # Simulate price with some noise and correlation to features
        cls.y = (
            100 * cls.X['square_feet'] +
            20000 * cls.X['bedrooms'] +
            30000 * cls.X['bathrooms'] +
            500 * (2023 - cls.X['year_built']) -
            1000 * cls.X['distance_to_city_center'] +
            np.random.normal(0, 50000, n_samples)
        )

        # Split the data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def setUp(self):
        """
        Create a fresh pipeline for each test
        """
        self.pipeline = HousePricePredictionPipeline()

    def test_pipeline_creation(self):
        """
        Test that the pipeline is created correctly
        """
        self.assertIsNotNone(self.pipeline.pipeline)
        self.assertEqual(len(self.pipeline.pipeline.steps), 2)

    def test_data_preparation(self):
        """
        Test data preparation method
        """
        X, y = self.pipeline.prepare_data(self.X_train, self.y_train)

        # Check feature names are stored
        self.assertIsNotNone(self.pipeline.feature_names)
        self.assertEqual(len(self.pipeline.feature_names), self.X_train.shape[1])

    def test_training(self):
        """
        Test that the pipeline can be trained without errors
        """
        try:
            self.pipeline.train(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")

    def test_prediction(self):
        """
        Test prediction functionality
        """
        # Train the pipeline first
        self.pipeline.train(self.X_train, self.y_train)

        # Make predictions
        predictions = self.pipeline.predict(self.X_test)

        # Check predictions
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_performance_metrics(self):
        """
        Test performance metrics calculation
        """
        # Train the pipeline
        self.pipeline.train(self.X_train, self.y_train)

        # Predict
        predictions = self.pipeline.predict(self.X_test)

        # Calculate metrics
        metrics = self.pipeline.get_metrics(self.y_test, predictions)

        # Print metrics for debugging
        print("Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        # Assert metrics are reasonable
        self.assertLess(metrics['rmse'], 100000)  # Root Mean Squared Error
        self.assertGreater(metrics['r2_score'], 0.5)  # R-squared score

    def test_feature_importance(self):
        """
        Test feature importance (for Ridge regression)
        """
        # Train the pipeline
        self.pipeline.train(self.X_train, self.y_train)

        # Get the Ridge model
        ridge_model = self.pipeline.pipeline.named_steps['regressor']

        # Check coefficients
        coefficients = ridge_model.coef_
        self.assertEqual(len(coefficients), self.X_train.shape[1])

        # Ensure coefficients are finite
        self.assertTrue(np.all(np.isfinite(coefficients)))

    def test_outlier_handling(self):
        """
        Test pipeline's handling of potential outliers
        """
        # Create a copy of training data with an extreme outlier
        X_with_outlier = self.X_train.copy()
        y_with_outlier = self.y_train.copy()

        # Add an extreme outlier
        X_with_outlier.loc[len(X_with_outlier)] = [10000, 10, 5, 1900, 0]
        y_with_outlier.loc[len(y_with_outlier)] = 5000000

        # Train on data with outlier
        try:
            self.pipeline.train(X_with_outlier, y_with_outlier)
            predictions = self.pipeline.predict(self.X_test)

            # Ensure predictions are still reasonable
            self.assertTrue(np.all(np.isfinite(predictions)))
        except Exception as e:
            self.fail(f"Failed to handle outlier with error: {e}")

    def test_prediction_single_sample(self):
        """
        Test prediction for a single house
        """
        # Train the pipeline
        self.pipeline.train(self.X_train, self.y_train)

        # Predict for a single sample
        single_sample = self.X_test.iloc[[0]]
        prediction = self.pipeline.predict(single_sample)

        # Check single prediction
        self.assertEqual(len(prediction), 1)
        self.assertTrue(np.isfinite(prediction[0]))

if __name__ == '__main__':
    unittest.main()
