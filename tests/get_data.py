import json

def load_test_dataset(file1, file2):
    try:
        # Load JSON data from both files
        with open(file1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        
        with open(file2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # Dictionary to store generated data
        generated_data_dict = {}
        for item in data2:
            if not item:  # Skip empty strings
                continue
            try:
                # Each item is already a JSON string that needs to be parsed
                response_data = json.loads(item)
                
                if response_data["num"] not in generated_data_dict:
                    generated_data_dict[response_data["num"]] = {
                        "rating": response_data["rating"],
                        "review": response_data["review"],
                        "genres": response_data.get("genres", [])
                    }
                
            except json.JSONDecodeError:
                #print(f"Skipping invalid JSON in response")
                pass
            except KeyError as e:
                print(f"Missing required field in response: {e}")
        
        output = []
        # Filter data1 to only include movies that have generated data
        for movie in data1:
            if movie["num"] not in generated_data_dict:
                continue
                
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
                    "original_rating": original_rating,
                    "generated_rating": generated_data_dict[movie["num"]]["rating"],  # We know this exists
                    "original_genres": original_genres,
                    "generated_genres": generated_data_dict[movie["num"]]["genres"],  # We know this exists
                    "original_reviews": original_reviews,
                    "generated_review": generated_data_dict[movie["num"]]["review"]   # We know this exists
                }
                
                output.append(movie_data)
                
            except json.JSONDecodeError:
                print(f"Skipping movie num {movie['num']} due to invalid reviews format.")
            except KeyError as e:
                print(f"Missing required field in movie data: {e}")
        
        print(f"Processed {len(output)} movies with generated data out of {len(data1)} total movies")
        return output

    except Exception as e:
        print(f"Error processing files: {e}")
        return None

def load_test_dataset2(file1, file2):
    try:
        # Load JSON data from both files
        with open(file1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        
        with open(file2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # Dictionary to store generated data
        generated_data_dict = {}
        for item in data2:
            if not item:
                continue
            
            try:
                # NEW: Assume `item` is already a dict
                response_data = item

                if response_data["num"] not in generated_data_dict:
                    generated_data_dict[response_data["num"]] = {
                        "rating": response_data["rating"],
                        "review": response_data["review"],
                        "genres": response_data.get("genres", [])
                    }

            except KeyError as e:
                print(f"Missing required field in response: {e}")
        
        output = []
        # Filter data1 to only include movies that have generated data
        for movie in data1:
            if movie["num"] not in generated_data_dict:
                continue

            try:
                reviews_data = json.loads(movie["reviews"])
                original_rating = reviews_data["rating"]
                original_reviews = [review for review in reviews_data["reviews"]]

                original_genres = movie.get("genres", "")
                if isinstance(original_genres, str):
                    original_genres = [genre.strip() for genre in original_genres.split(",") if genre.strip()]
                elif not isinstance(original_genres, list):
                    original_genres = []

                movie_data = {
                    "num": movie["num"],
                    "name": movie["name"],
                    "original_rating": original_rating,
                    "generated_rating": generated_data_dict[movie["num"]]["rating"],
                    "original_genres": original_genres,
                    "generated_genres": generated_data_dict[movie["num"]]["genres"],
                    "original_reviews": original_reviews,
                    "generated_review": generated_data_dict[movie["num"]]["review"]
                }

                output.append(movie_data)

            except json.JSONDecodeError:
                print(f"Skipping movie num {movie['num']} due to invalid reviews format.")
            except KeyError as e:
                print(f"Missing required field in movie data: {e}")

        print(f"Processed {len(output)} movies with generated data out of {len(data1)} total movies")
        return output

    except Exception as e:
        print(f"Error processing files: {e}")
        return None

