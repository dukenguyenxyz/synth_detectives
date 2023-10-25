# SynthDetectives@ALTA2023 Stacking the Odds: Transformer-based Ensemble for AI-generated Text Detection

Stacking ensemble of Transformers trained to detect AI-generated text for the [ALTA Shared Task 2023](https://www.alta.asn.au/events/sharedtask2023/).

"Our approach is novel in terms of its choice of models in that we use accessible and lightweight models in the ensemble. We show that ensembling the models results in an improved accuracy in comparison with using them individually. Our approach achieves an accuracy score of 0.9555 on the official test data provided by the shared task organisers."

## Directory structure

- `dataset`: dataset
- `src`: code

### Dataset

The dataset is provided by ALTA Shared Task 2023 on [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/14327)

- [training.json](dataset/training.json) - 18k evenly split human/machine generated training set with labels
- [validation_data.json](dataset/validation_data.json) - 2k validation set without labels
- [validation_sample_output.json](dataset/validation_sample_output.json) - 2k dummy validation output for output formatting reference
- [test_data.json](dataset/test_data.json) - 2k testing set used for leaderboard scoring on CodaLab

### Software

- [helper.py](src/helper.py) - helpers for EDA and model files
- [model.py](src/model.py) - model architecture and dataloading
- [eda.ipynb](src/eda.ipynb) - EDA notebook (all cells are preloaded)

---

- [build_embeddings.py](src/build_embeddings.py) - build and save embeddings for each Transformer model on the training set
- [train_weak_learners.py](src/train_weak_learners.py) - train the weak learners on the embeddings
- [train_meta_learner.py](src/train_meta_learner.py) - train the meta-learner on the weak learner predictions of the dataset embeddings
- [inference.py](src/inference.py) - perform inference using the ensemble

## System Requirement

The training was done on `python >= 3.8.10` on Google Cloud Platform's Vertex Colab GPU for GCE usage on NVIDIA A100 (40 GB). It was also previously tested with GeForce RTX 3060 on WSL2 Ubuntu. The configurations which are detailed below will work out-of-the-box for NVIDIA A100 (40 GB). However, for less performant GPUs, the `BATCH_SIZE` will need to be decreased. All adjustable parameters are recorded as constants at the top of the model files, specifically you ca change the `BATCH_SIZE` and `NUM_EPOCH` in [train_weak_learners.py](src/train_weak_learners.py) and [train_meta_learner.py](src/train_meta_learner.py).

## Installation

- Run `pip install -r requirements.txt`

## Training

- Ensure `training.json` exists in `dataset` folder.
- Run `pip build_embeddings.py` to build `[CLS]` embedding of the last hidden layer for the dataset using all Transformers (ALBERT, ELECTRA, RoBERTa, XLNet). If your GPU is not great, please reduce the `load_batch_size` in `src/model.py:TransformerModel.dataset`. This will produce the embeddings `.pt` file, `pretrained--dev=False--model=MODEL.pt`, for each of the Transformer `MODEL` variants above.
- Run `pip train_weak_learners.py` to train each of the Transformer weak learner using the previously produced embeddings. This will save the best weights for each weak learner in the following location `lightning_logs/version_VERSION/checkpoints/model=MODEL--dev=False--epoch=EPOCH-step=STEP--val_loss=VAL_LOSS.ckpt`. 
- Update the `checkpoints` array in [train_meta_learner.py](src/train_meta_learner.py) with the best weight path of each weak learner which was produced in the previous step. Note that the checkpoints have to follow the following order: `ALBERT, ELECTRA, RoBERTa, XLNet`.
- Run `pip train_meta_learner.py` to train the meta-learner Logistic Regression classifier using the best weights of the weak learners. This will save the best weight of the meta-learner.

## Inference

- Ensure `test_data.json` exists in `dataset` folder.
- Ensure you have the weights for each of the weak learner and the meta-learner from the training step.
- Update the `checkpoints` array (for the weak learners) and `lr_checkpoint_path` (for the meta-learner) in [inference.py](src/inference.py)
- Run `pip inference.py`. This will produce `answer.json` which contains the inference output.

## Authors

- [Duke Nguyen](https://github.com/dukeraphaelng)
- [Khaing Myat Noe Naing](https://github.com/KhaingNaing)
- [Aditya Joshi](https://scholar.google.com/citations?user=SbYRrvgAAAAJ&hl=en)

## License

- [MIT](LICENSE)