import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SpamEmailClassificationPipeline:
    def __init__(self):
        """
        Initialize the spam email classification pipeline
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', MultinomialNB())
        ])

    def train(self, X_train, y_train):
        """
        Train the classification pipeline

        :param X_train: Training email texts
        :param y_train: Training labels (0 for not spam, 1 for spam)
        """
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict spam for given emails

        :param X_test: Email texts to classify
        :return: Predictions (0 or 1)
        """
        return self.pipeline.predict(X_test)

    def get_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics

        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Dictionary of performance metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }

class TestSpamEmailClassificationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up test data for the entire test class
        """
        # Sample spam and non-spam emails
        spam_emails = [
            "FREE VIAGRA NOW! CLICK HERE!!!",
            "CONGRATULATIONS! You've won a million dollars!",
            "Urgent! Exclusive offer only for you today!",
            "GET RICH QUICK WITH THIS AMAZING OPPORTUNITY"
        ]

        non_spam_emails = [
            "Hey, can we discuss the project meeting tomorrow?",
            "Please find attached the quarterly report",
            "Reminder: Team lunch this Friday",
            "Minutes from last week's project review"
        ]

        # Create labels
        cls.X = spam_emails + non_spam_emails
        cls.y = [1] * len(spam_emails) + [0] * len(non_spam_emails)

        # Split the data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=42
        )

    def setUp(self):
        """
        Create a fresh pipeline for each test
        """
        self.pipeline = SpamEmailClassificationPipeline()

    def test_pipeline_creation(self):
        """
        Test that the pipeline is created correctly
        """
        self.assertIsNotNone(self.pipeline.pipeline)
        self.assertEqual(len(self.pipeline.pipeline.steps), 2)

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
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

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

        # Assert metrics are within reasonable ranges
        self.assertGreaterEqual(metrics['accuracy'], 0.5)
        self.assertLessEqual(metrics['accuracy'], 1.0)

        self.assertGreaterEqual(metrics['precision'], 0.0)
        self.assertLessEqual(metrics['precision'], 1.0)

        self.assertGreaterEqual(metrics['recall'], 0.0)
        self.assertLessEqual(metrics['recall'], 1.0)

        self.assertGreaterEqual(metrics['f1_score'], 0.0)
        self.assertLessEqual(metrics['f1_score'], 1.0)

    def test_spam_detection(self):
        """
        Test specific spam detection scenarios
        """
        # Train the pipeline
        self.pipeline.train(self.X_train, self.y_train)

        # Test specific spam and non-spam examples
        spam_test = [
            "WIN BIG MONEY NOW!!!",
            "EXCLUSIVE OFFER - CLICK HERE IMMEDIATELY"
        ]
        non_spam_test = [
            "Can we schedule a meeting?",
            "Here's the report you requested"
        ]

        # Predict spam for test cases
        spam_predictions = self.pipeline.predict(spam_test)
        non_spam_predictions = self.pipeline.predict(non_spam_test)

        # Assert spam emails are classified as spam (1)
        self.assertTrue(all(pred == 1 for pred in spam_predictions))

        # Assert non-spam emails are classified as non-spam (0)
        self.assertTrue(all(pred == 0 for pred in non_spam_predictions))

    def test_empty_input(self):
        """
        Test handling of empty input
        """
        # Train the pipeline
        self.pipeline.train(self.X_train, self.y_train)

        # Test empty list prediction
        empty_input = []
        predictions = self.pipeline.predict(empty_input)

        # Should return an empty list
        self.assertEqual(len(predictions), 0)

if __name__ == '__main__':
    unittest.main()
