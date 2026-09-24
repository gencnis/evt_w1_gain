
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
