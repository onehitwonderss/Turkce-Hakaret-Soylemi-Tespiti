from transformers import AutoTokenizer
from transformers import XLMRobertaForSequenceClassification, AutoModelForSequenceClassification
from transformers import pipeline
from options import Options
import argparse

options = Options()

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--text', help='Tahmin için bir metin girin', required=True)
args = parser.parse_args()

if options.model_name == "mdeberta-v3-base":
	model = AutoModelForSequenceClassification.from_pretrained(options.model_save_path)
elif options.model_name == "xlm-roberta-base":
	model = XLMRobertaForSequenceClassification.from_pretrained(options.model_save_path)
else:
	print('Geçersiz Model İsmi')


id2label = {0:'INSULT', 1:'OTHER', 2:'PROFANITY', 3:'RACIST', 4:'SEXIST'}
tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_save_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def predict(text):
	result = id2label[int(classifier(text)[0]['label'].split('_')[-1])]
	return result

if __name__ == '__main__':
	result = predict(args.text)
	print(result)