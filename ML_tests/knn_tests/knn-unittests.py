import sys
import unittest
import numpy as np
sys.path.append("ML/algorithms/knn")
from knn import KNearestNeighbor

class Test_KNN(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.KNearestNeighbor = KNearestNeighbor(2)
        self.KNearestNeighbor.train(self.X , self.y)
        
    def test_K_value(self):
        self.assertTrue(self.KNearestNeighbor.k==2)
    
    def test_predict_Loop0(self):
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict(test, num_loops=0)
        self.assertTrue(y_pred==0.0)
    
    def test_predict_Loop1(self):
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict(test, num_loops=1)
        self.assertTrue(y_pred==0.0)
    
    def test_predict_Loop2(self):
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict(test, num_loops=2)
        self.assertTrue(y_pred==0.0)
    
    def test_compute_distance_vectorized(self):
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict_labels(
            self.KNearestNeighbor.compute_distance_vectorized(test)
        )
        self.assertTrue(y_pred==0.0)
    
    def test_compute_distance_one_loop(self):
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict_labels(
            self.KNearestNeighbor.compute_distance_one_loop(test)
        )
        self.assertTrue(y_pred==0.0) 
    
    def test_compute_distance_two_loops(self):
        test = np.array([[4, 3]])
        y_pred = self.KNearestNeighbor.predict_labels(
            self.KNearestNeighbor.compute_distance_two_loops(test)
        )
        self.assertTrue(y_pred==0.0) 

if __name__ == "__main__":
    print("Running KNN Tests:")
    unittest.main()
    unittest.main()
