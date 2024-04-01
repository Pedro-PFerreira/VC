import json
from colorama import Fore, Style

json_test = None
brick_error_acc = 0
color_error_acc = 0
n_tests = 0

with open("./json/test.json", "r") as file:
    json_test = json.load(file)
    file.close()


def test(filename, num_bricks, num_colors):
    global brick_error_acc, color_error_acc, n_tests

    for test_case in json_test[
        "results"
    ]:  # Access the "results" key in the JSON object
        if test_case["file_name"] == filename or test_case["file_name"] in filename:
            print("\n\nTesting: " + filename)
            print("=" * 30)

            if test_case["num_detections"] == num_bricks:
                print(Fore.GREEN + "Number of detections test passed" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Number of detections test failed" + Style.RESET_ALL)

            brick_error_acc += (
                abs(test_case["num_detections"] - num_bricks)
                / test_case["num_detections"]
            )

            print(
                f"\tExpected number of bricks: {test_case['num_detections']}\n"
                f"\tDetected number of bricks: {num_bricks}\n"
            )

            if test_case["num_colors"] == num_colors:
                print(Fore.GREEN + "Number of colors test passed" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Number of colors test failed" + Style.RESET_ALL)

            color_error_acc += (
                abs(test_case["num_colors"] - num_colors) / test_case["num_colors"]
            )

            print(
                f"\tExpected number of colors: {test_case['num_colors']}\n"
                f"\tDetected number of colors: {num_colors}\n"
            )

    n_tests += 1


def compute_testing_score():
    print("=" * 30)
    print("Testing Summary")
    print("=" * 30)
    print(f"Brick Error Accumulation: {brick_error_acc / n_tests}")
    print(f"Color Error Accumulation: {color_error_acc / n_tests}")
    print("=" * 30)
