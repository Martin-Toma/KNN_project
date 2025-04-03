import json

def load_test_dataset(file1, file2):
    try:
        # Load JSON data from both files
        with open(file1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        
        with open(file2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # Dictionary to store generated data by movie num
        generated_data_dict = {}
        for item in data2:
            try:
                response_data = json.loads(item["response"])
                
                if item["num"] not in generated_data_dict:
                    generated_data_dict[item["num"]] = {
                        "rating": response_data["rating"],  # Store single rating
                        "review": response_data["review"],
                        "genres": set()  # Use a set to avoid duplicates
                    }
                
                # Store generated genres if available
                if "genres" in response_data:
                    generated_data_dict[item["num"]]["genres"].update(response_data["genres"])
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in response for movie num: {item['num']}")
        
        output = []
        for movie in data1:
            try:
                # Parse original reviews and ratings
                reviews_data = json.loads(movie["reviews"])
                original_rating = reviews_data["rating"]
                original_reviews = [review for review in reviews_data["reviews"]]
                
                # Ensure original genres are in list format
                original_genres = movie.get("genres", "")
                if isinstance(original_genres, str):
                    original_genres = [genre.strip() for genre in original_genres.split(",") if genre.strip()]
                elif not isinstance(original_genres, list):
                    original_genres = []
                
                movie_data = {
                    "num": movie["num"],
                    "name": movie["name"],
                    "original_rating": original_rating,  # Store original rating separately
                    "generated_rating": generated_data_dict.get(movie["num"], {}).get("rating", None),  # Single generated rating
                    "original_genres": original_genres,
                    "generated_genres": list(generated_data_dict.get(movie["num"], {}).get("genres", [])),
                    "original_reviews": original_reviews,
                    "generated_review": generated_data_dict.get(movie["num"], {}).get("review", {})  # Single generated review
                }
                
                output.append(movie_data)
                
            except json.JSONDecodeError:
                print(f"Skipping movie num {movie['num']} due to invalid reviews format.")
        
        return output

    except Exception as e:
        print(f"Error processing files: {e}")
        return None
