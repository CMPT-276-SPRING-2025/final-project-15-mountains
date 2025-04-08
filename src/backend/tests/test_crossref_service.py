import unittest
from unittest.mock import patch, MagicMock
# Assuming CrossRefService is importable from app and uses requests internally
# Adjust the import path if necessary
from app import CrossRefService, Study # Assuming Study is also defined/importable from app
import requests # Import requests to raise HTTPError

class TestCrossRefService(unittest.TestCase):
    def setUp(self):
        # Provide a dummy email, assuming it's needed like OpenAlexService
        self.service = CrossRefService(email="test@example.com")

    @patch('app.requests.get') # Patch requests.get within the 'app' module where CrossRefService uses it
    def test_search_works_by_keyword_success(self, mock_get): # Renamed test method for clarity
        """Test successful search and parsing of CrossRef data."""
        # Configure the mock response with 2 items
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/xyz123",
                        "title": ["Test Title 1"],
                        "author": [{"given": "Jane", "family": "Doe"}],
                        "created": {"date-parts": [[2023, 1, 15]]},
                        "abstract": "<jats:p>Abstract content</jats:p>",
                        "is-referenced-by-count": 10
                    },
                     {
                        "DOI": "10.1000/abc456",
                        "title": ["Test Title 2"],
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        # Call the search method
        query = "test query"
        result = self.service.search_works_by_keyword(query)

        # Assertions - expect raw dictionary response
        mock_get.assert_called_once() 
        # Check that the service returns the raw API response
        self.assertIsInstance(result, dict)
        self.assertIn('message', result)
        self.assertIn('items', result['message'])
        
        # Verify there are 2 items in the response
        items = result['message']['items']
        self.assertEqual(len(items), 2)
        
        # Check details of first item
        self.assertEqual(items[0]['DOI'], "10.1000/xyz123") 
        self.assertEqual(items[0]['title'][0], "Test Title 1")
        self.assertEqual(items[0]['author'][0]['given'], "Jane")
        self.assertEqual(items[0]['author'][0]['family'], "Doe")
        self.assertIn('abstract', items[0])
        self.assertEqual(items[0]['is-referenced-by-count'], 10)
        
        # Check second item has expected DOI and title
        self.assertEqual(items[1]['DOI'], "10.1000/abc456")
        self.assertEqual(items[1]['title'][0], "Test Title 2")

    @patch('app.requests.get')
    def test_search_works_by_keyword_api_error(self, mock_get): # Renamed test method
        """Test handling of API error response."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")
        mock_get.return_value = mock_response

        # Call the search method
        studies = self.service.search_works_by_keyword("error query")
        # Assert that None is returned on error (based on previous failure)
        self.assertIsNone(studies)

    # Add more tests: handling connection errors, empty results, different data formats etc.

if __name__ == '__main__':
    unittest.main() 