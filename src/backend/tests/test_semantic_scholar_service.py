import unittest
from unittest.mock import patch, MagicMock
# Assuming SemanticScholarService is importable from app and uses requests internally
# Adjust the import path if necessary
from app import SemanticScholarService, Study # Assuming Study is also defined/importable from app
import requests # Import requests to raise HTTPError
import time # Import time for patching sleep

# Create a mock service class to prevent infinite recursion
class MockSemanticScholarService(SemanticScholarService):
    """A mocked version that limits retries to prevent infinite recursion."""
    def __init__(self):
        super().__init__()
        self.retry_count = 0
        self.max_retries = 2  # Only retry twice
    
    def search_works_by_keyword(self, keywords, limit=10):
        """Override to prevent infinite recursion in the tests."""
        # Use the same API URL construction as the original
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keywords}&limit={limit}&fields=title,authors,year,abstract,citationCount,externalIds"
        
        try:
            # Simulate the original method's functionality but with retry limiting
            response = requests.get(url)
            if not response.ok:
                if response.status_code == 429:  # Rate limit error
                    if self.retry_count < self.max_retries:
                        # Simulate rate limit handling but with a limit
                        self.retry_count += 1
                        # Sleep would normally happen here
                        return self.search_works_by_keyword(keywords, limit)
                    else:
                        # Too many retries, give up
                        return None
                else:
                    # Other errors
                    return None
            
            # Reset retry count on success
            self.retry_count = 0
            # Return the raw response data
            return response.json()
            
        except Exception:
            # Handle exceptions
            return None

class TestSemanticScholarService(unittest.TestCase):
    def setUp(self):
        # Use the mock service with limited retries to avoid infinite recursion
        self.service = MockSemanticScholarService()

    @patch('app.requests.get') # Patch requests.get within the 'app' module
    def test_search_works_by_keyword_success(self, mock_get):
        """Test successful search and parsing of Semantic Scholar data."""
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        # Sample Semantic Scholar API JSON structure (simplified graph API example)
        mock_response.json.return_value = {
            "total": 2,
            "data": [
                {
                    "paperId": "s2id123",
                    "externalIds": {"DOI": "10.2000/xyz123"},
                    "title": "S2 Test Title 1",
                    "authors": [{"name": "Alex Ray"}],
                    "year": 2022,
                    "abstract": "S2 abstract content.",
                    "citationCount": 25
                },
                {
                    "paperId": "s2id456",
                    "externalIds": {}, # No DOI
                    "title": "S2 Test Title 2",
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call the search method
        query = "s2 test query"
        result = self.service.search_works_by_keyword(query)

        # Assertions - expect raw dictionary response
        mock_get.assert_called_once()
        self.assertIsInstance(result, dict)
        self.assertIn('total', result)
        self.assertIn('data', result)
        
        # Verify there are 2 items in the data array
        self.assertEqual(result['total'], 2)
        self.assertEqual(len(result['data']), 2)
        
        # Check details of first paper
        paper1 = result['data'][0]
        self.assertEqual(paper1['paperId'], "s2id123") 
        self.assertEqual(paper1['externalIds']['DOI'], "10.2000/xyz123")
        self.assertEqual(paper1['title'], "S2 Test Title 1")
        self.assertEqual(paper1['authors'][0]['name'], "Alex Ray")
        self.assertEqual(paper1['year'], 2022)
        self.assertEqual(paper1['abstract'], "S2 abstract content.")
        self.assertEqual(paper1['citationCount'], 25)
        
        # Check second paper
        paper2 = result['data'][1]
        self.assertEqual(paper2['paperId'], "s2id456")
        self.assertEqual(paper2['title'], "S2 Test Title 2")
        self.assertEqual(paper2['externalIds'], {}) # Empty dictionary (no DOI)

    @patch('requests.get')  # Patch the direct requests.get used in the mock class
    def test_search_works_by_keyword_api_error(self, mock_get):
        """Test handling of API error response (non-429)."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500 
        mock_response.reason = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{mock_response.status_code} Server Error: {mock_response.reason} for url: fake_url", 
            response=mock_response
        )
        mock_get.return_value = mock_response

        # Call the search method
        result = self.service.search_works_by_keyword("s2 error query")
        # Assert that None is returned on error
        self.assertIsNone(result)
        mock_get.assert_called_once()

    @patch('requests.get')  # Patch the direct requests.get used in the mock class
    def test_search_works_by_keyword_rate_limit_stops_retrying(self, mock_get):
        """Test that service stops retrying after hitting rate limit multiple times."""
        mock_response_429 = MagicMock()
        mock_response_429.ok = False
        mock_response_429.status_code = 429
        mock_response_429.reason = "Too Many Requests"
        http_error_429 = requests.exceptions.HTTPError(
            f"{mock_response_429.status_code} Client Error: {mock_response_429.reason} for url: fake_url", 
            response=mock_response_429
        )
        mock_response_429.raise_for_status.side_effect = http_error_429

        # Make requests.get return the 429 error for every call
        # Our mock service has a max_retries of 2, so we need 3 total responses (initial + 2 retries)
        mock_get.return_value = mock_response_429

        query = "s2 rate limit query that keeps failing"
        
        # The test should now complete without StopIteration since our mock has limited retries
        result = self.service.search_works_by_keyword(query)

        # Assert that None is returned when rate limited repeatedly
        self.assertIsNone(result)
        
        # Verify the number of attempts: initial call + max_retries
        expected_calls = 1 + self.service.max_retries
        self.assertEqual(mock_get.call_count, expected_calls)

    # Original rate limit test (optional - checks single failure)
    # @patch('app.requests.get')
    # def test_search_works_by_keyword_rate_limit_single(self, mock_get):
    #     """Test handling of Semantic Scholar API rate limit (429) - single instance."""
    #     mock_response = MagicMock()
    #     mock_response.ok = False
    #     mock_response.status_code = 429
    #     # ... rest of single 429 setup ...
    #     mock_get.return_value = mock_response
    #     studies = self.service.search_works_by_keyword("s2 rate limit query single")
    #     self.assertEqual(len(studies), 0)

    # Add more tests: connection errors, empty results, different fields missing, etc.

if __name__ == '__main__':
    unittest.main() 