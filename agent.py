import os
import time
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlmAgent:
    def __init__(self):
        # In a real environment, you would verify the API key here
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = "gemini-2.5-flash-preview-09-2025"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def run(self, user_message):
        """
        Main entry point for the agent to process a user message.
        """
        try:
            return self._generate_content(user_message)
        except Exception as e:
            logger.error(f"Error in agent run: {e}")
            return "I apologize, but I encountered an error processing your request."

    def _generate_content(self, prompt, retries=5):
        """
        Generates content using the Gemini API with exponential backoff.
        """
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        headers = {
            "Content-Type": "application/json"
        }

        delay = 1
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract text safely
                    try:
                        return result['candidates'][0]['content']['parts'][0]['text']
                    except (KeyError, IndexError):
                        logger.error(f"Unexpected response structure: {result}")
                        return "I received an unexpected response structure from the model."
                
                # If we get here, the status code was not 200
                logger.warning(f"Attempt {attempt + 1} failed with status {response.status_code}: {response.text}")
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed with exception: {e}")

            # Exponential backoff
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
        
        logger.error("All retries failed.")
        return "I am currently unable to reach the language model. Please try again later."

# Example usage for testing
if __name__ == "__main__":
    agent = LlmAgent()
    print(agent.run("Hello, are you working?"))