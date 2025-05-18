##Running Tests

To succesfully run tests a test_subset.json file is needed in the test_dataset folder, due to size constrains the file has been removed from the handed over script, but can be found on github.

To run tests run the test.py file, inside you can choose which dataset the file takes in.


##Test folder

- `test.py`: main testing file
- `test_genres.py`: helper testing file
- `test_reviews.py`: helper testing file
- `test_scores.py`: helper testing file
- `get_data.py`: helper testing file
- `all_results.txt`: combined file with results from all models
- `test_resulrs`: folder where test result are saved
- `perplexities`: folder with all perplexities saved
- `perplexities/calc_avg.py`: python file which calculates the avrage perplexity based on the json file
- `test_dataset`: folder with dataset containing human written reviews, ratings and ganres
- `new_datasets`: datasets with responses from all models
