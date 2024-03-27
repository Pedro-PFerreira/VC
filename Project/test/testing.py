import json
from colorama import Fore, Style

json_test = None

with open("./json/test.json", "r") as file:
    json_test = json.load(file)

def test(filename, num_bricks, num_colors):
    for test_case in json_test["results"]:  # Access the "results" key in the JSON object
        if test_case["file_name"] == filename or test_case["file_name"] in filename:
            print("\n\nTesting: " + filename)
            print("=" * 30)
            
            if test_case["num_detections"] == num_bricks:
                print(Fore.GREEN + "Number of detections test passed" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Number of detections test failed" + Style.RESET_ALL)

            print(
                f"\tExpected number of bricks: {num_bricks}\n"
                f"\tDetected number of bricks: {test_case['num_detections']}\n"
            )

            if test_case["num_colors"] == num_colors:
                print(Fore.GREEN + "Number of colors test passed" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Number of colors test failed" + Style.RESET_ALL)

            print(
                f"\tExpected number of colors: {num_colors}\n"
                f"\tDetected number of colors: {test_case['num_colors']}\n"
            )