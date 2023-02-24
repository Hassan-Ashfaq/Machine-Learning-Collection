import sys
import unittest
import numpy as np

sys.path.append("ML/algorithms/logisticregression")

from logistic_regression import LogisticRegression

class Test_Logistic_Regression(unittest.TestCase):
    def setUp(self):
        self.X1 = np.array([[ 9.1206996 , -4.13125487]])
        self.y = np.array(([1]))
        self.weights = np.zeros((self.X1.shape[1], 1))
        self.LogisticReg = LogisticRegression(self.X1, learning_rate=0.2, num_iters=100)
        
    def test_learningRate(self):
        self.assertTrue(self.LogisticReg.lr, 0.2)
    
    def test_num_iters(self):
        self.assertTrue(self.LogisticReg.num_iters, 100)
        
    def test_weights_matrix(self):
        w, b = self.LogisticReg.train(self.X1, self.y)
        self.assertTrue(self.weights.shape==w.shape)
    
    def test_sigmoid(self):
        out = self.LogisticReg.sigmoid(0)
        self.assertTrue(out==0.5)
 
if __name__ == "__main__":
    print("Running Logistic Regression Tests:")
    unittest.main()
