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

```bash
pip install - requirements.txt
```
