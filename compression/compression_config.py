from pathlib import Path

COMPRESSION_OUTPUT_DIR = 'compression/saved_compressed'
# # for loading the models - state dicts (later will be a dict so learning rate and accuracy measure will be extracted from it as well)
# best_covidbert_dict_path = 
# best_bertweet_dict_path = 

# # assuming model_dict in path is a pt file that looks like this:
# model_dict = {
#     "state_dict": model.state_dict(),
#     "lr_rate": lr_rate,
#     "best_acc": best_acc
# }

# SAVE_COMPRESSED_MODELS_DIR = # directory with 2 folders - with model key ["bertweet" or "covidbert"]
