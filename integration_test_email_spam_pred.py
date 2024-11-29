import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

class SpamClassificationSystem:
    def __init__(self):
        """
        Initialize the spam classification pipeline
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', MultinomialNB())
        ])
    
    def train(self, X_train, y_train):
        """
        Train the spam classification model
        
        :param X_train: Training email texts
        :param y_train: Training labels (0 for ham, 1 for spam)
        """
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, emails):
        """
        Predict spam or ham for given emails
        
        :param emails: List of email texts
        :return: Predictions (0 for ham, 1 for spam)
        """
        return self.pipeline.predict(emails)
    
    def predict_proba(self, emails):
        """
        Get spam probability for given emails
        
        :param emails: List of email texts
        :return: Probability of being spam
        """
        return self.pipeline.predict_proba(emails)[:, 1]

def load_sample_email_dataset():
    """
    Generate a sample email dataset for testing
    
    :return: Tuple of (emails, labels)
    """
    ham_emails = [
        "Meeting scheduled for tomorrow",
        "Please review the project report",
        "Invitation to quarterly team meeting",
        "Your bank statement is ready",
        "Confirmation of your recent order"
    ]
    
    spam_emails = [
        "URGENT: You've won a million dollars!",
        "FREE VIAGRA CHEAP ONLINE",
        "Increase your wealth in 24 hours",
        "CONGRATULATIONS! Click here to claim prize",
        "Hot singles in your area want to meet you"
    ]
    
    emails = ham_emails + spam_emails
    labels = [0] * len(ham_emails) + [1] * len(spam_emails)
    
    return emails, labels

@pytest.fixture
def spam_classifier():
    """
    Fixture to create and train a spam classifier
    """
    # Load dataset
    emails, labels = load_sample_email_dataset()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.3, random_state=42
    )
    
    # Create and train classifier
    classifier = SpamClassificationSystem()
    classifier.train(X_train, y_train)
    
    return {
        'classifier': classifier,
        'X_test': X_test,
        'y_test': y_test
    }

def test_spam_classification_accuracy(spam_classifier):
    """
    Integration test to verify spam classification performance
    """
    classifier = spam_classifier['classifier']
    X_test = spam_classifier['X_test']
    y_test = spam_classifier['y_test']
    
    # Make predictions
    predictions = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Check if accuracy meets minimum threshold
    assert accuracy >= 0.7, f"Accuracy {accuracy} is below acceptable threshold"

def test_spam_probability_threshold(spam_classifier):
    """
    Test spam probability calculation
    """
    classifier = spam_classifier['classifier']
    X_test = spam_classifier['X_test']
    
    # Get spam probabilities
    spam_probas = classifier.predict_proba(X_test)
    
    # Check probability range
    assert np.all((spam_probas >= 0) & (spam_probas <= 1)), \
        "Spam probabilities must be between 0 and 1"

def test_specific_spam_detection(spam_classifier):
    """
    Test detection of specific spam-like emails
    """
    classifier = spam_classifier['classifier']
    
    test_emails = [
        "FREE OFFER CLICK NOW!!!",
        "Legitimate project update",
        "You've inherited $10 million",
        "Your monthly bank statement"
    ]
    
    expected_labels = [1, 0, 1, 0]  # spam or not spam
    
    predictions = classifier.predict(test_emails)
    
    # Check specific email classifications
    assert np.array_equal(predictions, expected_labels), \
        "Failed to correctly classify specific test emails"

# Optional: Detailed error analysis
def test_confusion_matrix(spam_classifier):
    """
    Generate and analyze confusion matrix
    """
    classifier = spam_classifier['classifier']
    X_test = spam_classifier['X_test']
    y_test = spam_classifier['y_test']
    
    predictions = classifier.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Optional assertions on confusion matrix
    assert cm[1, 1] > 0, "No spam emails correctly identified"
    assert cm[0, 0] > 0, "No ham emails correctly identified"
