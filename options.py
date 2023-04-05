class Options:
	model_name = "mdeberta-v3-base" # or xlm-roberta-base
	max_seq_len = 128
	learning_rate = 2e-5
	epochs = 8
	batch_size = 4
	data_source = "dataset.csv"
	data_source_2 = None
	model_save_path = "./models/" + model_name + "/model/"
	tokenizer_save_path = "./models/" + model_name + "/model/"
