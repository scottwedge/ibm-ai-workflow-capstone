import app
import requests
import unittest
import pandas as pd
from ast import literal_eval

'''
response = app.put('http://localhost:5000/api/v1/user/create',
    data=json.dumps({"username": "test", "password": "abc"}),
    content_type='application/json')
'''
# API
class TestAPI(unittest.TestCase):
    def test_connection(self):
        print('Test connection')
        response = requests.get("http://127.0.0.1:5000")
        self.assertEqual(response.status_code, 200)

    def test_connection_train(self):
        print('Test train')
        response = requests.get("http://127.0.0.1:5000/train")
        self.assertEqual(response.status_code, 200)
        
    def test_connection_predict(self):
        print('Test predict')
        response = requests.get("http://127.0.0.1:5000/predict")
        self.assertEqual(response.status_code, 200)
    

class TestModel(unittest.TestCase):
    def test_model_predict(self):
        print('Test model predict')
        ## create some new data
        X_new_data = {}
        X_new_data['customer_id'] = [52, 24, 1]
        X_new_data['invoice'] = [78, 29, 1]
        X_new_data['price'] = [3.02, 2.88, 2.08]
        X_new_data['revenue'] = [14749.57, 5843.79, 722.72]
        X_new_data['stream_id'] = [724, 497, 52]
        X_new_data['num_streams'] = [7277, 2533, 493]
        X_new_data['revenue_past3days'] = [53922, 15875, 2109]
        X_new_data['revenue_past7days'] = [100209, 89002, 2639]
        X_new_data['revenue_past30days'] = [321764, 487572, 12981]
        X_new = pd.DataFrame(X_new_data)
        X_new.head()

        query = X_new.to_dict()

        response = requests.get("http://127.0.0.1:5000/train", json = query)
        response = literal_eval(response.text)

        self.assertEqual(len(response), 3)


if __name__ == "__main__":
    unittest.main()