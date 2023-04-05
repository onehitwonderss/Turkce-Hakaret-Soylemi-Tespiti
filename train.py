# -*- coding: utf-8 -*-

import re
import time
import numpy as np
import pandas as pd
import nltk as nlp
import torch
import string
import torch.nn as nn
import nltk # Natural Language toolkit
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from options import Options


options = Options()

print(options.model_name)

if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('GPU is:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



nltk.download("stopwords")  #downloading stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download("stopwords")


def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    text = re.sub(r'[^ \w\.]', '', text)
    text=nltk.word_tokenize(text)
    text =[word for word in text if not word in set(stopwords.words("turkish") + ['mi', 'bi'])]
    lemma=nlp.WordNetLemmatizer()
    text=[lemma.lemmatize(word) for word in text]
    text=" ".join(text)
    return text

def read_csv(path, path2 = None):
    df = pd.read_csv(path, encoding='utf-8', delimiter='|')
    print(dict(df.groupby("target").count()["id"]))
    if path2== None:
        df['clean_text'] = df.text.apply(lambda x: clean_text(x))
        return df
    df_z = pd.read_csv(path2, encoding='utf-8', delimiter='|')
    non = pd.DataFrame(df.values.repeat(df.target=='OTHER', axis=0), columns=df.columns)
    df_z['offansive'][58] = 1
    df_z.target = df_z.target.apply(lambda x: x.strip())
    df_z.offansive = df_z.offansive.apply(lambda x: int(x))
    dict(df_z.groupby("target").count()["id"])
    df_z.columns = ['id','text','is_offensive','target']
    df = df.drop('id', axis=1)
    df_z = df_z.drop('id', axis=1)
    frames = [df, df_z, non]
    result = pd.concat(frames)
    df = result
    df['clean_text'] = df.text.apply(lambda x: clean_text(x))
    return df



def set_tokenizer(df):
    if options.model_name == "mdeberta-v3-base":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    elif options.model_name == "xlm-roberta-base":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenized_feature_raw = tokenizer.batch_encode_plus(
                                # Sentences to encode
                                df.clean_text.values.tolist(), 
                                # Add '[CLS]' and '[SEP]'
                                add_special_tokens = True      
                       )
    # collect tokenized sentence length 
    token_sentence_length = [len(x) for x in tokenized_feature_raw['input_ids']]

    features = df.clean_text.values.tolist()
    target = df.target.values.tolist()

    MAX_LEN = options.max_seq_len
    tokenized_feature = tokenizer.batch_encode_plus(
                                # Sentences to encode
                                features, 
                                # Add '[CLS]' and '[SEP]'
                                add_special_tokens = True,
                                # Add empty tokens if len(text)<MAX_LEN
                                padding = 'max_length',
                                # Truncate all sentences to max length
                                truncation=True,
                                # Set the maximum length
                                max_length = MAX_LEN, 
                                # Return attention mask
                                return_attention_mask = True,
                                # Return pytorch tensors
                                return_tensors = 'pt'       
                       )
    return tokenizer, tokenized_feature


def get_train_val_data_loader(tokenized_feature, target):
    le = LabelEncoder()
    le.fit(target)
    target_num = le.transform(target)


    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(tokenized_feature['input_ids'], 
                                                                                                             target_num,
                                                                                                                    tokenized_feature['attention_mask'],
                                                                                                      random_state=2018, test_size=0.2, stratify=target)

    batch_size = options.batch_size
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our test set
    validation_data = TensorDataset(validation_inputs, validation_masks, torch.tensor(validation_labels))
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader, le


def get_model(number_of_classes, len_token_embeddings):
    if options.model_name == "mdeberta-v3-base":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/mdeberta-v3-base", num_labels = number_of_classes)
    elif options.model_name == "xlm-roberta-base":
        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base", 
            # Specify number of classes
            num_labels = len(set(target)), 
            # Whether the model returns attentions weights
            output_attentions = False,
            # Whether the model returns all hidden-states 
            output_hidden_states = False
        )
    model.resize_token_embeddings(len_token_embeddings)
    if device == torch.device("cuda"):
        model.cuda()
    return model



def train(model, train_dataloader):
    # Optimizer & Learning Rate Scheduler
    optimizer = AdamW(model.parameters(),
                      lr = options.learning_rate, 
                      eps = 1e-8 
                    )

    criteria = nn.CrossEntropyLoss()

    epochs = options.epochs
    total_steps = len(train_dataloader) * epochs


    loss_values = []
    print('total steps per epoch: ',  len(train_dataloader) / options.batch_size)
    # looping over epochs
    for epoch_i in range(0, epochs):
        
        print('training on epoch: ', epoch_i)
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                print('training on step: ', step)
                print('total time used is: {0:.2f} s'.format(time.time() - t0))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids.long(),
                            token_type_ids=None,
                            attention_mask=b_input_mask.long(),
                            labels=b_labels.long())
            # get loss
            logits = outputs.logits
            loss = criteria(logits, b_labels.long())
            loss.backward()
            # total loss
            total_loss += loss.item()
            # update optimizer
            optimizer.step()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        final_prediction = np.argmax(logits, axis=-1).flatten()
        print(final_prediction, b_labels)
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("average training loss: {0:.2f}".format(avg_train_loss))
    return model


def validate(model, validation_dataloader, le):
    t0 = time.time()
    model.eval()
    predictions,true_labels =[],[]
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        final_prediction = np.argmax(logits, axis=-1).flatten()
        predictions.append(final_prediction)
        true_labels.append(label_ids)
        
    print('total time used is: {0:.2f} s'.format(time.time() - t0))

    # convert numeric label to string
    final_prediction_list = le.inverse_transform(np.concatenate(predictions))
    final_truelabel_list = le.inverse_transform(np.concatenate(true_labels))


    cr = classification_report(final_truelabel_list, 
                               final_prediction_list, 
                               output_dict=False)
    print(cr)


def predict_one_sample(model, text):
    id2label = {0:'INSULT', 1:'OTHER', 2:'PROFANITY', 3:'RACIST', 4:'SEXIST'}

    model  =model.to('cpu')

    from transformers import pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print(id2label[int(classifier(text)[0]['label'].split('_')[-1])])


if __name__ == '__main__':
    data_frame = read_csv(options.data_source, options.data_source_2)
    features = data_frame.clean_text.values.tolist()
    target = data_frame.target.values.tolist()
    tokenizer, tokenized_feature = set_tokenizer(data_frame)
    train_dataloader, validation_dataloader, le = get_train_val_data_loader(tokenized_feature, target)
    model = get_model(len(target), len(tokenizer))
    model = train(model, train_dataloader)
    validate(model, validation_dataloader, le)
    predict_one_sample(model, "Bu bir test cümlesidir.")
    tokenizer.save_pretrained(options.model_save_path)
    model.save_pretrained(options.tokenizer_save_path)
