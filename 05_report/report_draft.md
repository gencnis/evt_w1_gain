
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
