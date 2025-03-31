import json

# Combines data from two JSON files containing movie information and reviews,
def load_test_dataset(file1, file2):
    try:
        # Load the JSON data from both files
        with open(file1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        
        with open(file2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # Create a dictionary to store generated reviews by movie num for faster lookup
        generated_reviews_dict = {}
        for item in data2:
            #print(f"Processing review for movie num: {item['num']}")
            if item["num"] not in generated_reviews_dict:
                generated_reviews_dict[item["num"]] = []
            generated_reviews_dict[item["num"]].append({
                "text": json.loads(item["response"])["review"],
                "rating": json.loads(item["response"])["rating"],
                "idx": item["idx"]68030
            })
        
        output = []
        for movie in data1:
            movie_data = {
                "num": movie["num"],
                "name": movie["name"],
                "original_reviews": [],
                "generated_reviews": generated_reviews_dict.get(movie["num"], [])
            }
            
            # Process original reviews
            original_reviews = json.loads(movie["reviews"])
            for review in original_reviews:
                movie_data["original_reviews"].append({
                    "text": review["text"],
                    "rating": review["rating"]
                })
            
            output.append(movie_data)
        
        return output
    
    except Exception as e:
        print(f"Error processing files: {e}")
        return None