import unittest
import requests
import json

class TestAPIEndpoint(unittest.TestCase):
    def setUp(self):
        """Set up test case with base URL and headers"""
        self.base_url = "http://localhost:9696/predict"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.test_data = {
            'ph': 7.875895135481787,
            'hardness': 226.28478781681216,
            'solids': 12710.249451611751,
            'chloramines': 7.303126583151656,
            'sulfate': 346.4032581373675,
            'conductivity': 445.3741474328599,
            'organic_carbon': 6.063461912529135,
            'trihalomethanes': 63.12804403102223,
            'turbidity': 4.238589203481521
        }

    def test_post_endpoint(self):
        """Test POST request returns 200 status code"""
        # Make POST request
        response = requests.post(
            url=self.base_url,
            headers=self.headers,
            data=json.dumps(self.test_data)
        )
        
        # Assert response status code is 200
        self.assertEqual(response.status_code, 200, 
            f"Expected status code 200 but got {response.status_code}")
        
        print(json.dumps(response.json(), indent=4))


if __name__ == "__main__":
    unittest.main()