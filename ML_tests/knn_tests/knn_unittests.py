"""Module unittest knn algo."""
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/knn")

# If run from local:
# sys.path.append('../../ML/algorithms/knn/')
from knn import KNearestNeighbor

class TestKNN(unittest.TestCase):
    """
    Test KNN Algo

    ...

    Attributes
    ----------
    x : int 
    y : int
    KNearestNeighbor : model
    """
    def setUp(self):
        '''SetUp'''
        self.x = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.KNearestNeighbor = KNearestNeighbor(2)
        self.KNearestNeighbor.train(self.x , self.y)
        
    def test_k_value(self):
        '''test_k_value'''
        self.assertTrue(self.KNearestNeighbor.k==2)
    
    def test_predict_loop0(self):
        '''test_predict_loop0'''
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict(test, num_loops=0)
        self.assertTrue(y_pred==0.0)
    
    def test_predict_loop1(self):
        '''test predict loop1'''
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict(test, num_loops=1)
        self.assertTrue(y_pred==0.0)
    
    def test_predict_loop2(self):
        '''test predict loop2'''
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict(test, num_loops=2)
        self.assertTrue(y_pred==0.0)
    
    def test_compute_distance_vectorized(self):
        '''test compute distance vectorized'''
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict_labels(
            self.KNearestNeighbor.compute_distance_vectorized(test)
        )
        self.assertTrue(y_pred==0.0)
    
    def test_compute_distance_one_loop(self):
        '''test compute distance one loop'''
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict_labels(
            self.KNearestNeighbor.compute_distance_one_loop(test)
        )
        self.assertTrue(y_pred==0.0) 
    
    def test_compute_distance_two_loops(self):
        '''test compute distance two loop'''
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict_labels(
            self.KNearestNeighbor.compute_distance_two_loops(test)
        )
        self.assertTrue(y_pred==0.0) 

if __name__ == "__main__":
    print("Running KNN Tests:")
    unittest.main()