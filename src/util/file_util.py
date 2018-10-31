import json
import pickle
def load_json(json_path):
    with open(json_path, 'r') as f:
        answer_dict = json.load(f)
        return answer_dict

def dump_pickle(file_name, entity):
    with open(file_name, 'wb') as f:
        pickle.dump(entity, f)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        ret = pickle.load(f)
    return ret