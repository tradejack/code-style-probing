PY150K_DIR = "data/py150"
PY150K_CODE_DIR = "data/py150/py150_files"
PY150K_TRAIN_AST = "data/py150/python100k_train.json"
PY150K_EVAL_AST = "data/py150/python50k_eval.json"
PY150K_TRAIN_CODE = "data/py150/py150_files/python100k_train.txt"
PY150K_EVAL_CODE = "data/py150/py150_files/python50k_eval.txt"

PLBART_TRAIN = "datasets/plbart_train.hf"
PLBART_TEST = "datasets/plbart_test.hf"

ARCH = "plbart_token_level_embed"
BATCH_SIZE = 8
NUM_EPOCHS = 100

NAME = f"{ARCH}_batch_{BATCH_SIZE}"
