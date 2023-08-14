import json
import pandas as pd

def load_dataset(path, test=True):
    '''Convert samples in JSON to dataframe
    0 if the text is AI-generated
    1 if the text is human-generated
    '''
    data = []
    columns = ['id', 'text', 'label']
    with open(path) as f:
        lines = f.readlines()        
        if test:
            for line in lines:
                line_dict = json.loads(line)
                data.append([line_dict['id'], line_dict['text'], line_dict['label']])
        else:
            columns = columns[:-1]
            for line in lines:
                line_dict = json.loads(line)
                data.append([line_dict['id'], line_dict['text']])

    return pd.DataFrame(data, columns=columns).set_index('id')