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
