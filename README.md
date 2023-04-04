# Teknofest 2023 Türkçe Hakaret Söylemi Tespiti

![hummingbird](https://user-images.githubusercontent.com/33387838/229937474-caa41f78-2169-4a5e-b50a-cde4fd1ee466.png)
        
Bu uygulamada türkçe metinlerde geçen hakaret içerikleri 5 farklı etiket değerinde sınıflandırılmıştır. Sınıflandırma yapılırken xlm-roberta-base ve mdeberta-v3 modelleri kullanılmıştır. Sonuç olarak ~%94 F1 skoru ile 'INSULT', 'SEXIST', 'PROFANITY', 'RACIST' ve 'OTHER' sınıfları sınıflandırılmıştır.

# Ön Gereksinimler
Çalışma ortamının hazırlanması için miniconda ile sanal Python 3.8 ortamı oluşturulmasını ve CUDA driverlerının yüklü olmasını öneriyoruz. 
```bash
conda create -n env_name python==3.8
```

# Kurulum
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

Çalışma kapsamında etiketlemiş olduğumuz 1000 adet etiketli veriyi <a href="https://drive.google.com/file/d/11s-T8R07C67UJZEx_Y66Xm7S9w5pfEbb/view?usp=share_link" target="_blank" > google drive </a> </p> linkinden indirebilirsiniz.

# Eğitim

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
Eğitime başlamadan önce train.py dosyasının bulunduğu dizinde models/model_name//model/ klasörlerini oluşturmanız gerekmektedir. Eğitim sonucu kat sayılar bu klasöre kayıt edilecektir.
