import os
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score
import csv



def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to calculate accuracy, F1 score, and recall
def calculate_metrics(data):
    y_true = []
    y_pred = []
    for entry in data:
        y_true.append(entry['label'])
        y_pred.append(entry['llm_judge'])
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return accuracy, f1, recall

# Function to print the results
def print_results(file_path, accuracy, f1, recall):
    print("\n" + "="*50)
    print(f"Results for JSON file: {file_path}")
    print("="*50)
    print(f"\nAccuracy : {accuracy:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(f"Recall   : {recall:.2f}")
    print("="*50 + "\n")


def remove_last_character_if_bracket(file_path):
    with open(file_path, 'rb+') as file:
        file.seek(-1, os.SEEK_END)
        if file.read(1) == b"]":
            file.seek(-1, os.SEEK_END)
            file.truncate()

def append_to_json_list(data_record, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        # Check if the file already contains data and hence, requires a comma
        file.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
        position = file.tell()  # Get the current position
        if position > 1:  # More than just the opening bracket
            file.write(",")  # Add a comma to separate the new record
        json.dump(data_record, file, ensure_ascii=False)  # Write the new data record

def finalize_json_list(file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write("]")

def save_to_file(args: dict, data_record: dict):
    path_name = "test_exps/" + args.judge_model + "/" + args.dataset + "/"
    file_name = f"{args.file_name}_start_data_index_{args.start_index}.json"
    file_path = os.path.join(path_name, file_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)
    # If the file doesn't exist, create it and write an opening bracket.
    # If it exists, remove the last character if it's a closing bracket.
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("[")
    else:
        remove_last_character_if_bracket(file_path)

    # Append the data_record to the file
    append_to_json_list(data_record, file_path)

    # After appending all records, finalize the JSON list with a closing bracket
    finalize_json_list(file_path)


def save_multi_agents_to_file(args: dict, data_record: dict):
    # path_name = "test_exps/" + args.judge_model + "/" + args.dataset + "/"
    # file_name = f"{args.file_name}_start_data_index_{args.start_index}.json"
    # file_path = os.path.join(path_name, file_name)
    # path_name =
    file_path = args.output_file
    # if not os.path.exists(path_name):
    #     os.makedirs(path_name)
    # If the file doesn't exist, create it and write an opening bracket.
    # If it exists, remove the last character if it's a closing bracket.
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("[")
    else:
        remove_last_character_if_bracket(file_path)

    # Append the data_record to the file
    append_to_json_list(data_record, file_path)

    # After appending all records, finalize the JSON list with a closing bracket
    finalize_json_list(file_path)

def load_csv(csv_path):
    """
    Load data from a CSV file and extract prompts and model responses.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing two lists: prompts and model_responses.
    """
    prompts = []
    model_responses = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                prompts.append(row["goal"])
                model_responses.append(row["target"])
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {csv_path} was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected column in CSV: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV file: {e}")

    return prompts, model_responses

def load_json(json_path):
    """
    Load data from a JSON file and extract user inputs and model responses.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        tuple: A tuple containing two lists: user_inputs and model_responses.
    """
    user_inputs = []
    model_responses = []

    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for entry in data:
                user_inputs.append(entry["user_input"])
                model_responses.append(entry["model_output"])
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_path} was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected key in JSON: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"The file {json_path} is not a valid JSON.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the JSON file: {e}")

    return user_inputs, model_responses

def load_data(file_path):
    """
    Load data from a file based on its extension and extract user inputs and model responses.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: A tuple containing two lists: inputs and responses.
    """
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':
        return load_csv(file_path)
    elif file_extension == '.json':
        return load_json(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}. Only .csv and .json are supported.")