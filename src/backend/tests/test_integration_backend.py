import unittest
# Import the Flask app instance from your main application file
from app import app

class TestFlaskAPIIntegration(unittest.TestCase):
    def setUp(self):
        # Create a test client for the Flask application
        self.client = app.test_client()
        # Enable testing mode. This disables error catching during request handling,
        # so that you get better error reports when performing test requests.
        app.testing = True

    def test_health_endpoint(self):
        """Test the /health endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")

    def test_verification_without_claim(self):
        """Test the verification endpoint with missing 'claim' data."""
        # Assuming your verification endpoint is '/api/verify_claim' and expects POST
        # Adjust endpoint path if necessary
        response = self.client.post('/api/verify_claim', json={}) # Sending empty JSON
        # Expecting a 400 Bad Request error because 'claim' is missing
        self.assertEqual(response.status_code, 400)
        # Optionally, check the error message if your API provides one
        # data = response.get_json()
        # self.assertIn("Missing 'claim' in request", data.get("error", ""))

if __name__ == '__main__':
    unittest.main() 