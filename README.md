# Jax-Flax-Turkiye
## Jax/Flax ile ilgili Turkçe Bilgiler ve son gelişmelerin paylaşılması amacıyla kurulmuştur.
### JAX Google Brain ekibi tarafından geliştirilen yeni bir sinir ağı kütüphanesidir.
JAX, Hızlandırılmış Doğrusal Cebir (XLA), Autograd ve Tam Zamanında Derleme (JIT) kullanmak üzere tasarlanmıştır.

XLA (Hızlandırılmış Doğrusal Cebir), potansiyel olarak kaynak kodu değişikliği olmadan modelleri hızlandırabilen doğrusal cebir için alana özgü bir derleyicidir. JAX, XLA kullanarak Numpy programlarını GPU ve TPU ile de işletilmesini sağlar, esas yeniliklerinden biri de budur.

Autograd, bize Numpy ve Python kodunu ayırt etmenin bir yolunu sağlayabilecek bir python kütüphanesidir, farklı işlemler için gradyanları bulunur. Sinir ağlarındaki özellikle ileri ve geri yayınlama (forward&back propagation) özelliğini sağlar.

Böylece JAX, bu özelliklerin birleşimiyle yüksek perfomans Makine Öğrenimi (ML) araştırmaları yapmanın imkanını sağlar.

JAX aynı zamanda fonksiyon derlemeleri sağlar (function transformations). Bunlar:
  - Paralelleştirme ( jax.pmap): Kodu birden çok hızlandırıcı arasında otomatik olarak paralel hale getirir (CPU, GPU ve TPU)
  - Autodiff ( jax.grad): temel aritmetik işlemlerden yararlanarak bir fonksiyonun türevini değerlendirmek için bir teknik.
  - JIT derlemesi ( jax.jit): Hızlandırıcı işlemleri (GPU ve TPU)
  - Otomatik vektörleme (jax.vmap)

#JAX örneği:

import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads

Daha fazla bilgi için : https://github.com/google/jax

@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}


### Flax, ilk olarak Google Brain Ekibi içindeki mühendisler ve araştırmacılar tarafından (JAX ekibiyle) geliştirilmeye başlandı ve şimdi open source topluluğuyla ortak geliştiriliyor

Flax, Felixibility ve JAX terimlerinin birleşmesinden oluşmuştur.
Flax, yüksek performans için tasarlanmış JAX tabanlı bir sinir ağı kütüphanesidir. Özellike Derin Öğrenme (DL) araştırmaları için tasarlanmıştır.

Genel olarak :
  - Flax, esneklik için tasarlanmış, JAX tabanlı yüksek performanslı bir sinir ağı kitaplığı ve ekosistemidir: Bir çerçeveye özellikler ekleyerek değil, bir örnek oluşturarak ve eğitim döngüsünü değiştirerek yeni eğitim biçimlerini deneyin.
  - Flax, JAX ekibiyle yakın işbirliği içinde geliştirilmektedir ve araştırmanıza başlamak için ihtiyacınız olan her şeyle birlikte gelir.
  - Sinir ağı API'si (flax.linen): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout
  - Yardımcı programlar ve modeller: çoğaltılmış eğitim, seri hale getirme ve kontrol noktası belirleme, metrikler, cihazda önceden getirme
  - Alışılmışın dışında çalışan eğitici örnekler: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging
  - Hızlı, ayarlanmış büyük ölçekli uçtan uca örnekler: CIFAR10, ImageNet üzerinde ResNet, Transformer LM1b
 
 Daha fazla bilgi için : https://github.com/google/flax
 
 
 
 @software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.6.2},
  year = {2020},
}

