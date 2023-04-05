# <p align="center">Teknofest 2023 Türkçe Hakaret Söylemi Tespiti </p>

<p align="center"><img id="radius_deneme" src="https://user-images.githubusercontent.com/33387838/229937474-caa41f78-2169-4a5e-b50a-cde4fd1ee466.png" width="30%" height="30%" align="center"/></a> </p>
        
Bu uygulamada türkçe metinlerde geçen hakaret içerikleri 5 farklı etiket değerinde sınıflandırılmıştır. Sınıflandırma yapılırken xlm-roberta-base ve mdeberta-v3 modelleri kullanılmıştır. Sonuç olarak ~%94 F1 skoru ile 'INSULT', 'SEXIST', 'PROFANITY', 'RACIST' ve 'OTHER' sınıfları sınıflandırılmıştır.

# Ön Gereksinimler
Çalışma ortamının hazırlanması için miniconda ile sanal Python 3.8 ortamı oluşturulmasını ve CUDA driverlerının yüklü olmasını öneriyoruz. 
```bash
conda create -n env_name python==3.8
```

# Kurulum 🎯
Ön gereksinimler sağlandıktan sonra aşağıdaki komut ile kurulumu gerçekleştirebilirsiniz.

 
Gerekli Kütüphane | Version
------------ | -------------
torch | 2.0.0
torchvision | 0.15.1
transformers | 4.18.0
numpy | <1.24.0
sentencepiece | 0.1.96
nltk
pandas
scikit-learn

```bash
pip install -r requirements.txt
```

# Veri Kümesi

Çalışma kapsamında etiketlemiş olduğumuz 1000 adet etiketli veriyi <a href="https://drive.google.com/file/d/1lKaEpcC3FXlFSaLAfAOZ1_b6dbKrjRWL/view?usp=share_link" target="_blank" > google drive </a> linkinden indirebilirsiniz.

# Eğitim 🚀

Model eğitmek için veri setinin path değerini options.py'da göstermeniz geremektedir. Ardından options.py dosyasında hyperparametreleri düzenleyerek eğitim denemesi yapabilirsiniz.

```python

class Options:
	model_name = "mdeberta-v3-base" # or xlm-roberta-base
	max_seq_len = 128
	learning_rate = 2e-5
	epochs = 1
	batch_size = 4
	data_source = "dataset.csv"
	data_source_2 = None
	model_save_path = "./models/" + model_name + "/model/"
	tokenizer_save_path = "./models/" + model_name + "/model/"
```
Model seçmek için options.py dosyasıdan model_name değişkenini "mdeberta-v3-base" ya da "xlm-roberta-base" olarak belirtmeniz yeterlidir.
Eğitime başlamadan önce train.py dosyasının bulunduğu dizinde models/model_name//model/ klasörlerini oluşturmanız gerekmektedir. Eğitim sonucu kat sayılar bu klasöre kayıt edilecektir. Options parametlerini ayarladıktan sonra ve dataset pathini doğru girdiğinizden emin olduktan sonra aşağıdaki script ile eğitimi başlatabilirsiniz.

```bash
python train.py
```

Eğitim T4 GPU'da ortalama 30 dakika sürmektedir.

# Pre-trained Modeller

Önceden eğitilmiş modelleri kullanmak için <a href= "https://drive.google.com/file/d/1-0hPfWsupW5y5nLZEzwSfp73RG3jA_km/view?usp=share_link" target="_blank" > mdeberta-v3 </a> ve <a href= "https://drive.google.com/file/d/1LqarxkF2wSWetbS5WQs6SxF95gwsO-Nx/view?usp=share_link" target="_blank" > xlm-roberta-base </a> linklerini kullanabilirsiniz. Modeller kullanmak için zip dosyalarını ./models/model_name (mdeberta-v3 ya da xlm-roberta-base)/model/ altına açmanız gerekmektedir.

# Nasıl Kullanılır? 

Eğitilmiş modeli predict_one_text.py scriptindeki örnek ile kullanabilirsiniz.
```python
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
```

# DEMO!

Önceten eğitmiş olduğumuz modelleri aşağıdaki linklerden deneyin.

<a href= "https://huggingface.co/spaces/CanKorkut/finetune-mdeberta-v3-for-turkish-hatespeech-classification" target="_blank" > mdeberta-v3 </a>
<a href= "https://huggingface.co/spaces/CanKorkut/turkish-hatespeech-detection" target="_blank" > xlm-roberta-base </a>

# Ekip Üyeleri

[Can Korkut](https://github.com/CanKorkut)
[Zeynep Çay](https://github.com/zynpcy)
