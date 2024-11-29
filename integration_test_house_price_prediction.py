import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class HousePriceIntegrationTestSystem:
    def __init__(self):
        """
        Initialize the complete house price prediction system
        """
        # Define preprocessing for numeric and categorical columns
        numeric_features = ['square_feet', 'lot_size', 'year_built', 'total_rooms']
        categorical_features = ['neighborhood', 'property_type']
        
        # Preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Full pipeline with preprocessor and model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Tracking for system-wide metrics
        self.training_metrics = {}
        self.validation_metrics = {}
    
    def load_data(self, data_path):
        """
        Load data from a CSV file
        
        :param data_path: Path to the data file
        :return: Features and target variables
        """
        try:
            df = pd.read_csv(data_path)
            
            # Validate required columns
            required_columns = ['square_feet', 'lot_size', 'year_built', 'total_rooms', 
                                'neighborhood', 'property_type', 'price']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Split features and target
            X = df[['square_feet', 'lot_size', 'year_built', 'total_rooms', 
                    'neighborhood', 'property_type']]
            y = df['price']
            
            return X, y
        except Exception as e:
            raise IOError(f"Error loading data: {e}")
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """
        Train the model and generate performance metrics
        
        :param X: Feature matrix
        :param y: Target prices
        :param test_size: Proportion of data for testing
        :return: Performance metrics dictionary
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        training_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2_score': r2_score(y_train, y_train_pred)
        }
        
        validation_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2_score': r2_score(y_test, y_test_pred)
        }
        
        # Store metrics
        self.training_metrics = training_metrics
        self.validation_metrics = validation_metrics
        
        return {
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics
        }
    
    def predict(self, X_new):
        """
        Make predictions for new data
        
        :param X_new: New feature data
        :return: Predicted prices
        """
        return self.pipeline.predict(X_new)

class IntegrationTestHousePricePredictionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment and generate synthetic data
        """
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic house data
        data = pd.DataFrame({
            'square_feet': np.random.normal(2000, 500, n_samples),
            'lot_size': np.random.normal(5000, 1000, n_samples),
            'year_built': np.random.randint(1950, 2023, n_samples),
            'total_rooms': np.random.randint(2, 10, n_samples),
            'neighborhood': np.random.choice(['suburban', 'urban', 'rural'], n_samples),
            'property_type': np.random.choice(['single_family', 'condo', 'townhouse'], n_samples),
            'price': None  # Will be calculated
        })
        
        # Generate synthetic price based on features
        data['price'] = (
            200 * data['square_feet'] + 
            50 * data['lot_size'] + 
            1000 * (2023 - data['year_built']) +
            20000 * data['total_rooms'] +
            np.random.normal(0, 50000, n_samples)
        )
        
        # Save synthetic data
        cls.synthetic_data_path = 'synthetic_house_data.csv'
        data.to_csv(cls.synthetic_data_path, index=False)
    
    def setUp(self):
        """
        Initialize the system before each test
        """
        self.system = HousePriceIntegrationTestSystem()
    
    def test_data_loading(self):
        """
        Test data loading functionality
        """
        X, y = self.system.load_data(self.synthetic_data_path)
        
        # Validate loaded data
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
    
    def test_model_training(self):
        """
        Test end-to-end model training and evaluation
        """
        # Load data
        X, y = self.system.load_data(self.synthetic_data_path)
        
        # Train and evaluate
        results = self.system.train_and_evaluate(X, y)
        
        # Validate training results
        self.assertIn('training_metrics', results)
        self.assertIn('validation_metrics', results)
        
        # Check metrics
        training_metrics = results['training_metrics']
        validation_metrics = results['validation_metrics']
        
        # Validate MAE (Mean Absolute Error)
        self.assertLess(training_metrics['mae'], 100000)
        self.assertLess(validation_metrics['mae'], 100000)
        
        # Validate R2 Score
        self.assertGreater(training_metrics['r2_score'], 0.5)
        self.assertGreater(validation_metrics['r2_score'], 0.5)
    
    def test_prediction_pipeline(self):
        """
        Test the complete prediction pipeline
        """
        # Load and train on data
        X, y = self.system.load_data(self.synthetic_data_path)
        self.system.train_and_evaluate(X, y)
        
        # Create new sample data for prediction
        new_data = pd.DataFrame({
            'square_feet': [2500],
            'lot_size': [6000],
            'year_built': [2010],
            'total_rooms': [4],
            'neighborhood': ['suburban'],
            'property_type': ['single_family']
        })
        
        # Make prediction
        prediction = self.system.predict(new_data)
        
        # Validate prediction
        self.assertEqual(len(prediction), 1)
        self.assertIsInstance(prediction[0], (int, float))
        self.assertGreater(prediction[0], 0)
    
    def test_system_robustness(self):
        """
        Test system's ability to handle various input scenarios
        """
        # Load and train on data
        X, y = self.system.load_data(self.synthetic_data_path)
        self.system.train_and_evaluate(X, y)
        
        # Test scenarios
        test_scenarios = [
            # Missing one optional feature
            pd.DataFrame({
                'square_feet': [2500],
                'lot_size': [6000],
                'year_built': [2010],
                'total_rooms': [4],
                'neighborhood': ['suburban']
            }),
            # Extreme feature values
            pd.DataFrame({
                'square_feet': [10000],
                'lot_size': [50000],
                'year_built': [1900],
                'total_rooms': [20],
                'neighborhood': ['rural'],
                'property_type': ['single_family']
            })
        ]
        
        for scenario in test_scenarios:
            try:
                prediction = self.system.predict(scenario)
                self.assertEqual(len(prediction), 1)
                self.assertIsInstance(prediction[0], (int, float))
                self.assertGreater(prediction[0], 0)
            except Exception as e:
                self.fail(f"Prediction failed for scenario: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up synthetic data file after tests
        """
        if os.path.exists(cls.synthetic_data_path):
            os.remove(cls.synthetic_data_path)

if __name__ == '__main__':
    unittest.main()
