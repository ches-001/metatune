import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

@pytest.fixture
def dataset():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

def test_perceptron_classification(dataset):
    X, y = dataset
    model = Perceptron()
    model.fit(X, y)
    
    # Perform assertions to validate the model's behavior
    # For example:
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert all(y_pred == y)

