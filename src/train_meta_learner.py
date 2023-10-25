BATCH_SIZE = 128
NUM_EPOCH = 100

# Update the following checkpoints in the following order: albert, electra, roberta, xlnet
checkpoints = [
    'lightning_logs/version_0/checkpoints/model=albert--dev=True--epoch=2-step=60--val_loss=0.39.ckpt',
    'lightning_logs/version_1/checkpoints/model=electra--dev=True--epoch=2-step=60--val_loss=0.53.ckpt',
    'lightning_logs/version_4/checkpoints/model=roberta--dev=True--epoch=3-step=80--val_loss=0.63.ckpt',
    'lightning_logs/version_5/checkpoints/model=xlnet--dev=True--epoch=3-step=80--val_loss=0.62.ckpt'
]

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from helper import load_dataset
from model import TransformerModel, Data, get_dataloaders, SoftMaxLit

DEV = False
device = torch.cuda.current_device()
df = load_dataset('../dataset/training.json', test=True)

pretrained_datasets_x = [
    f"pretrained--dev={DEV}--model=albert.pt",
    f"pretrained--dev={DEV}--model=electra.pt",
    f"pretrained--dev={DEV}--model=roberta.pt",
    f"pretrained--dev={DEV}--model=xlnet.pt"
]

model_y_arr = []
for model_name, pretrained_dataset_x, ckpt in zip(list(TransformerModel.MODELS.keys()), pretrained_datasets_x, checkpoints):
    n_inputs = TransformerModel.MODELS[model_name]['dim']
    model = SoftMaxLit(n_inputs, 2).load_from_checkpoint(n_inputs=n_inputs, n_outputs=2, checkpoint_path=ckpt)
    x = torch.load(pretrained_dataset_x).to(device)
    y_hat = model(x)

    # Free up memory
    del x
    torch.cuda.empty_cache()
    y_first = y_hat

    model_y_arr.append(y_first)

lr_dataset_x = torch.cat(model_y_arr, dim=1).detach()

lr_dataset = Data(df, x=lr_dataset_x)
lr_dataloaders = get_dataloaders(lr_dataset, BATCH_SIZE)

lr_model = SoftMaxLit(lr_dataset_x.shape[1], 2)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor='val_loss',
    mode='min',
    filename=f'model=lr--dev={DEV}'
)

trainer = pl.Trainer(callbacks = [checkpoint_callback], max_epochs=NUM_EPOCH) # callbacks=[checkpoint_callback]
trainer.fit(model=lr_model, train_dataloaders=lr_dataloaders['train'], val_dataloaders=lr_dataloaders['val'])
best_lr_model = lr_model.load_from_checkpoint(n_inputs=lr_dataset_x.shape[1], n_outputs=2, checkpoint_path=checkpoint_callback.best_model_path)
trainer.test(best_lr_model, dataloaders=lr_dataloaders['test'])