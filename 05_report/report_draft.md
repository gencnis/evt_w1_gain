
## Reproducibility Note: Difference Between the Paper and Official GAIN Code

GAIN'in makalede tanımlanan hint mekanizması ile yazarların yayımladığı resmi kod birebir aynı değildir.

Makalede hint vektörü teorik olarak:

`H = B ⊙ M + 0.5(1 - B)`

şeklinde tanımlanır. Bu yapıda `0.5`, discriminator'a ilgili bileşenin gözlenen mi yoksa tamamlanan mı olduğunun açıklanmadığı belirsiz konumu temsil eder. Makaledeki Algorithm 1'de discriminator kaybı da hint mekanizmasının belirsiz bıraktığı bileşenler üzerinden kurulmaktadır.

Yazarların resmi GAIN kodunda ise hint:

`H_mb = M_mb * H_mb_temp`

şeklinde oluşturulmaktadır. Bu uygulamada `0.5` kullanılmaz ve hint yalnızca gözlenen değişkenlerin bir bölümünü discriminator'a açıklar. Resmi implementasyonda discriminator kaybı da makaledeki Algorithm 1'den farklı olarak tüm bileşenler üzerinde hesaplanmaktadır.

Bu fark, resmi GitHub deposundaki "Hint mechanism different in the paper?" başlıklı Issue #2'de de tartışılmıştır. GAIN'in yazarlarından Jinsung Yoon, teorik makalede kullanılan hint yapısı ile pratik implementasyonun farklı olduğunu doğrulamış; pratikte maskenin yaklaşık %90'ının hint olarak verilmesinin tercih edildiğini belirtmiştir.

Bu nedenle bu çalışmada iki farklı kavram birbirinden ayrılacaktır:

1. Paper-oriented GAIN: Makaledeki Algorithm 1 ve teorik hint mekanizmasına daha yakın uygulama.
2. Author-code-oriented GAIN: Yazarların resmi GitHub implementasyonuna daha yakın uygulama.

Bu ayrım, deney sonuçlarının yorumlanması ve orijinal makaledeki sayısal sonuçların yeniden üretilebilirliğinin değerlendirilmesi açısından önemlidir.

### Deneysel Gözlem

Makale odaklı ve resmi kod odaklı iki GAIN varyantı aynı veri, eksiklik maskesi, eğitim tohumu ve hiperparametreler altında ayrıca karşılaştırılmıştır.

Makale odaklı E5 deneyinde RMSE 0.058253, resmi kod odaklı E7 deneyinde ise 0.058247 olarak ölçülmüştür.

İki sonucun pratik olarak aynı olması, hint ve loss uygulamasındaki dokümante edilmiş farkın bu deney düzeninde GAIN'in eksik veri tamamlama doğruluğu üzerinde belirgin bir etkisinin olmadığını göstermektedir.

Bu nedenle orijinal makalede Spam veri seti için raporlanan daha düşük RMSE ile bu çalışmadaki sonuçlar arasındaki farkın başka uygulama veya değerlendirme ayrıntılarından kaynaklanıp kaynaklanmadığı ayrıca incelenmelidir.

### Değerlendirme Protokolünün Etkisi

Yeniden üretilebilirlik incelemesinde yalnızca model mimarisi ve eğitim
algoritmasının değil, hata ölçütünün uygulanma biçiminin de sonuçları önemli
ölçüde etkileyebildiği görülmüştür.

E8 deneyinde aynı GAIN modeli ve aynı tamamlanmış değerler iki farklı RMSE
hesaplama yöntemiyle değerlendirilmiştir. Çalışmada başlangıçta kullanılan
normalize edilmiş veri uzayındaki değerlendirme 0.058247 RMSE verirken,
yazarların resmi GAIN kodundaki normalizasyon ve `rmse_loss` mantığına daha
yakın değerlendirme 0.054750 RMSE vermiştir.

Bu fark modelin daha iyi öğrenmesinden kaynaklanmamaktadır; iki değer aynı
tamamlanmış veri üzerinde hesaplanmıştır. Dolayısıyla GAIN sonuçlarını yeniden
üretirken değerlendirme protokolünün de model ve hiperparametreler kadar açık
biçimde tanımlanması gerekmektedir.

### Eğitim Süresinin Etkisi

Yazar kodundaki örnek yapılandırmaya yaklaşmak amacıyla eğitim süresi 2.000
iterasyondan 10.000 iterasyona çıkarılmıştır. Yazar koduna benzer değerlendirme
protokolünde RMSE 0.054750'den 0.054585'e düşmüştür.

Bu değişim sınırlıdır. Dolayısıyla mevcut deneyde orijinal makaledeki Spam
sonucuyla kalan farkın temel nedeninin eğitim iterasyonu olmadığı
değerlendirilmektedir. Buna karşılık değerlendirme ve normalizasyon
protokolünün RMSE üzerinde daha belirgin bir etkisi olduğu görülmüştür.

### Resmi Kodun Yeniden Üretilebilirlik Testi

Yazarların resmi GitHub implementasyonu ayrıca izole bir ortamda doğrudan
çalıştırılmıştır. Kullanılan depo sürümü
`ed53e6d0be14a8d4ce35eff46449d4047bcb483e` commit'idir.

Python 3.10, TensorFlow CPU 2.15.1 ve NumPy 1.26.4 ortamında resmi Spam veri
seti ve önerilen hiperparametrelerle yapılan iki ayrı 10.000 iterasyonluk
çalıştırmada RMSE değeri NaN olarak oluşmuştur.

Buna karşılık daha kısa tanısal çalıştırmalarda 500 iterasyonda 0.0549,
1.000 iterasyonda 0.0529 ve 5.000 iterasyonda 0.0524 gibi sonlu RMSE değerleri
elde edilmiştir.

Resmi kod rastgele tohumları sabitlemediği için bu sonuçlar eğitim iterasyonu
açısından doğrudan karşılaştırılabilir değildir. Her çalıştırmada eksiklik
maskesi, ağ başlangıç değerleri, mini-batch sırası ve gürültü örnekleri
değişmektedir. Bu nedenle 10.000 iterasyondaki NaN davranışını eğitim
süresinden ayırabilmek amacıyla rastgele tohumların sabitlendiği ek bir
yeniden üretilebilirlik testi gerçekleştirilmiştir.

### Rastgele Tohumun Yeniden Üretilebilirliğe Etkisi

Resmi implementasyondaki iki bağımsız 10.000 iterasyonluk çalıştırmanın NaN
üretmesi üzerine rastgelelik kaynakları sabitlenerek ek bir deney yapılmıştır.

Seed=42 altında 1.000, 2.000, 5.000 ve 10.000 iterasyon için sırasıyla 0.0564,
0.0548, 0.0541 ve 0.0533 RMSE elde edilmiştir. Böylece 10.000 iterasyonun
kendi başına sayısal bozulmaya yol açmadığı görülmüştür.

Bu bulgu, seed sabitlenmeyen resmi implementasyonda sonuçların koşudan koşuya
değişebildiğini ve iterasyon sayısı gibi değişkenlerin etkisini inceleyebilmek
için rastgelelik kaynaklarının kontrol edilmesi gerektiğini göstermektedir.

Orijinal GAIN makalesinde Spam veri seti için 0.0513 ± 0.0016 RMSE
raporlanmıştır. Seed=42 ile resmi implementasyonda 10.000 iterasyonda elde
edilen 0.0533 RMSE bu değerden daha yüksek olmakla birlikte, tek bir seed
sonucu olduğundan makaledeki çoklu deney ortalamasıyla doğrudan eşdeğer
olarak yorumlanmamıştır.

### Çoklu Tohum ile Resmi Kodun Yeniden Üretimi

Tek bir rastgele tohumdan elde edilen sonucun genellenebilirliğini
değerlendirmek amacıyla yazarların resmi GAIN implementasyonu 0-9 arasındaki
10 farklı seed ile çalıştırılmıştır. Her koşuda Spam veri seti, %20 eksiklik
oranı, 128 mini-batch büyüklüğü, 0.9 hint oranı, 100 alpha değeri ve 10.000
eğitim iterasyonu kullanılmıştır.

On koşunun sekizinde sonlu RMSE elde edilirken seed 4 ve seed 8 koşuları NaN
ile sonuçlanmıştır. Sonlu sekiz koşunun ortalama RMSE değeri 0.052797,
örneklem standart sapması ise 0.000909 olarak hesaplanmıştır. En düşük sonlu
RMSE seed 7 için 0.051160 olarak elde edilmiştir.

Böylece başarılı koşular için:

`0.0528 ± 0.0009`

sonucu elde edilmiştir.

Orijinal GAIN makalesinde Spam veri seti için `0.0513 ± 0.0016` RMSE
raporlanmıştır. Başarılı resmi kod koşuları makalede raporlanan sayısal
performansa yakın bir bölgede bulunmakla birlikte, 10 koşunun ikisinin NaN
üretmesi nedeniyle bu sonuç birebir bir yeniden üretim olarak
değerlendirilmemiştir.

Bu deney, yalnızca ortalama başarımın değil, farklı rastgele başlangıçlardaki
sayısal kararlılığın da yeniden üretilebilirlik değerlendirmesinde dikkate
alınması gerektiğini göstermektedir.

### MICE Yakınsama Duyarlılık Analizi

Final karşılaştırmada kullanılan MICE-style IterativeImputer, `max_iter=10`
ayarında tüm seed'lerde early-stopping yakınsama uyarısı üretmiştir. Bu nedenle
sonucun iterasyon sınırına duyarlı olup olmadığını değerlendirmek amacıyla aynı
10 maske üzerinde `max_iter=50` ile ek bir duyarlılık deneyi yapılmıştır.

Bu deneyde ortalama RMSE 0.053365, standart sapma 0.003485 ve ortalama çalışma
süresi 69.05 saniye olarak ölçülmüştür. On koşunun sekizinde yakınsama uyarısı
devam etmiştir.

Dolayısıyla iterasyon sınırının 10'dan 50'ye çıkarılması yakınsama sorununu
tutarlı biçimde çözmemiş, buna karşılık hata değişkenliğini ve hesaplama
maliyetini artırmıştır. Bu nedenle final karşılaştırmada baştan belirlenen
`max_iter=10` MICE-style baseline korunmuş, 50 iterasyonluk deney yalnızca
duyarlılık analizi olarak raporlanmıştır.
EOFcat >> 05_report/report_draft.md <<'EOF'

### MICE Yakınsama Duyarlılık Analizi

Final karşılaştırmada kullanılan MICE-style IterativeImputer, `max_iter=10`
ayarında tüm seed'lerde early-stopping yakınsama uyarısı üretmiştir. Bu nedenle
sonucun iterasyon sınırına duyarlı olup olmadığını değerlendirmek amacıyla aynı
10 maske üzerinde `max_iter=50` ile ek bir duyarlılık deneyi yapılmıştır.

Bu deneyde ortalama RMSE 0.053365, standart sapma 0.003485 ve ortalama çalışma
süresi 69.05 saniye olarak ölçülmüştür. On koşunun sekizinde yakınsama uyarısı
devam etmiştir.

Dolayısıyla iterasyon sınırının 10'dan 50'ye çıkarılması yakınsama sorununu
tutarlı biçimde çözmemiş, buna karşılık hata değişkenliğini ve hesaplama
maliyetini artırmıştır. Bu nedenle final karşılaştırmada baştan belirlenen
`max_iter=10` MICE-style baseline korunmuş, 50 iterasyonluk deney yalnızca
duyarlılık analizi olarak raporlanmıştır.

### Final KNN, MICE ve GAIN Karşılaştırması

Ana uygulama deneyinde UCI Spambase veri seti üzerinde 0-9 arasındaki 10 farklı
rastgele tohum kullanılmıştır. Her seed için verinin yaklaşık %20'si MCAR
mekanizmasıyla yapay olarak eksik hale getirilmiş ve aynı eksiklik maskesi KNN,
MICE-style IterativeImputer ve GAIN yöntemlerine uygulanmıştır. Böylece her üç
yöntem aynı gizlenmiş hücreler üzerinde değerlendirilmiştir.

| Yöntem | Ortalama RMSE | Standart Sapma | Min. RMSE | Maks. RMSE | Ortalama Süre |
|---|---:|---:|---:|---:|---:|
| KNN | 0.053333 | 0.000688 | 0.052437 | 0.054372 | 2.30 s |
| MICE-style | 0.051219 | 0.001292 | 0.048626 | 0.053134 | 8.66 s |
| GAIN | 0.052605 | 0.001112 | 0.050904 | 0.054280 | 20.17 s |

Bu deney düzeninde en düşük ortalama RMSE MICE-style yönteminde elde edilmiştir.
GAIN'in ortalama RMSE değeri KNN'den daha düşük olmakla birlikte GAIN üç yöntem
arasında en yüksek hesaplama süresine sahiptir.

MICE-style yöntemi max_iter=10 ayarında yakınsama uyarısı verdiğinden aynı 10
maske üzerinde max_iter=50 ile ek bir duyarlılık analizi yapılmıştır. Bu
deneyde ortalama RMSE 0.053365, standart sapma 0.003485 ve ortalama çalışma
süresi 69.05 saniye olarak ölçülmüş; 10 koşunun 8'inde yakınsama uyarısı devam
etmiştir. Bu nedenle final karşılaştırmada baştan belirlenen max_iter=10
baseline'ı korunmuştur.

Sonuçlar, tek bir veri seti, MCAR eksiklik mekanizması ve seçilen
hiperparametrelerle sınırlıdır. Bu nedenle yöntemlerden birinin genel olarak
diğerlerinden üstün olduğu sonucuna varılmamaktadır.
