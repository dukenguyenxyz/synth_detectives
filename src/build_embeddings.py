from model import TransformerModel
from helper import load_dataset

DEV = False
df = load_dataset('../dataset/training.json', test=True)

for cur_model_name in list(TransformerModel.MODELS.keys()):
    TransformerModel(cur_model_name).dataset(df, DEV, save=True, delete=True)