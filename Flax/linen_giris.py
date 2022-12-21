# Yuklemek ve İçe Aktarmak

# Son JAXLib versiyonu yuklemek:
!pip install --upgrade -q pip jax jaxlib
# Bastan Flax eklemek:
!pip install --upgrade -q git+https://github.com/google/flax.git
   
import functools
from typing import Any, Callable, Sequence, Optional
import jax
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn

# Basit Module Tanımlama

class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # list ve alt moduller ile ne yapmamız gerektigini biliyoruz
    self.layers = [nn.Dense(feat) for feat in self.features]
    # tek submodule için sadece şunu yazıyoruz:
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitMLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)

# MLP için @compact hali ile :
class SimpleMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      # bir isim vermek mumkun!
      # normal olarak şu şekilde olur "Dense_0", "Dense_1", ...
      # x = nn.Dense(feat)(x)
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleMLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)

# Değişkenleri kullanmak
class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init,  # RNG geçişi
                        (inputs.shape[-1], self.features))  # şekil bilgisi
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),)
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3)
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameters:\n', init_variables)
print('output:\n', y)

# ExplicitDense
class ExplicitDense(nn.Module):
  features_in: int  # <-- explicit input shape
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  def setup(self):
    self.kernel = self.param('kernel',
                             self.kernel_init,
                             (self.features_in, self.features))
    self.bias = self.param('bias', self.bias_init, (self.features,))

  def __call__(self, inputs):
    y = lax.dot_general(inputs, self.kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),)
    y = y + self.bias
    return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitDense(features_in=4, features=3)
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameters:\n', init_variables)
print('output:\n', y)

# Genel Değişkenler
class Counter(nn.Module):
  @nn.compact
  def __call__(self):
    # easy pattern to detect if we're initializing
    is_initialized = self.has_variable('counter', 'count')
    counter = self.variable('counter', 'count', lambda: jnp.zeros((), jnp.int32))
    if is_initialized:
      counter.value += 1
    return counter.value


key1 = random.PRNGKey(0)

model = Counter()
init_variables = model.init(key1)
print('initialized variables:\n', init_variables)

y, mutated_variables = model.apply(init_variables, mutable=['counter'])

print('mutated variables:\n', mutated_variables)
print('output:\n', y)
