import json
import re

# Simulated input string
text = '''
ppose we'll ever see them again. We may yet, Mr. Frodo. We may. Sam, I'm glad you're with me. THE END

### Response:
"{
  "rating": 8.9,
  "genres": ["Adventure", "Drama", "Fantasy"],
  "review": "An excellent production! Casting was perfect for realistic portrayal of these beloved characters,br scenery also translated well from the book."
}"
'''

# Extract the JSON string using regex
match = re.search(r'### Response:\s*"({.*?})"', text, re.DOTALL)
if match:
    json_str = match.group(1)
    
    # Fix any escaped double quotes or syntax errors if needed
    json_str = json_str.replace('\\"', '"')

    # Load JSON
    data = json.loads(json_str)

    # Access values by keys
    rating = data.get("rating")
    genres = data.get("genres")
    review = data.get("review")

    print("Rating:", rating)
    print("Genres:", genres)
    print("Review:", review)
else:
    print("JSON content not found.")