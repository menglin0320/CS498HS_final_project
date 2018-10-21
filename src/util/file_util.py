import json

def load_json(json_path):
    with open(json_path, 'r') as f:
        answer_dict = json.load(f)
        return answer_dict
