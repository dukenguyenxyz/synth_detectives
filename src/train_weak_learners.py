BATCH_SIZE = 128
NUM_EPOCH = 300

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from helper import load_dataset
from model import TransformerModel, Data, get_dataloaders, SoftMaxLit

DEV = False

df = load_dataset('../dataset/training.json', test=True)

# https://stackoverflow.com/questions/65445651/t5tokenizer-requires-the-sentencepiece-library-but-it-was-not-found-in-your-envicheckpoints = []
checkpoints = []
for cur_model_name in list(TransformerModel.MODELS.keys()):
    # cur_model_name
    cur_dataset_x = torch.load(f'pretrained--dev={DEV}--model={cur_model_name}.pt')
    cur_data = Data(df, x=cur_dataset_x)
    cur_dataloaders = get_dataloaders(cur_data, BATCH_SIZE)
    cur_model = SoftMaxLit(TransformerModel.MODELS[cur_model_name]['dim'], 2)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename=f'model={cur_model_name}--dev={DEV}' + '--{epoch}-{step}--{val_loss:.2f}'
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=NUM_EPOCH)
    trainer.fit(model=cur_model, train_dataloaders=cur_dataloaders['train'], val_dataloaders=cur_dataloaders['val'])

    checkpoints.append(checkpoint_callback.best_model_path)
    best_model = cur_model.load_from_checkpoint(n_inputs=TransformerModel.MODELS[cur_model_name]['dim'], n_outputs=2, checkpoint_path=checkpoint_callback.best_model_path)
    trainer.test(best_model, dataloaders=cur_dataloaders['test'])

    del cur_dataset_x
    del cur_data.x
    torch.cuda.empty_cache()