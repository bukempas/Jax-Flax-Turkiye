# JIT (Just in Time) Derleyicisi ile Flax module :

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      # JIT the Module (it's __call__ fn by default.)
      x = nn.jit(nn.Dense)(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(3), 2)
x = random.uniform(key1, (4,4))

model = MLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)

# Vmap (vectorization map) Derleyicisi ile module :
class RawDotProductAttention(nn.Module):
  attn_dropout_rate: float = 0.1
  train: bool = False

  @nn.compact
  def __call__(self, query, key, value, bias=None, dtype=jnp.float32):
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim

    n = query.ndim
    attn_weights = lax.dot_general(
        query, key,
        (((n-1,), (n - 1,)), ((), ())))
    if bias is not None:
      attn_weights += bias
    norm_dims = tuple(range(attn_weights.ndim // 2, attn_weights.ndim))
    attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
    attn_weights = nn.Dropout(self.attn_dropout_rate)(attn_weights,
                                                      deterministic=not self.train)
    attn_weights = attn_weights.astype(dtype)

    contract_dims = (
        tuple(range(n - 1, attn_weights.ndim)),
        tuple(range(0, n  - 1)))
    y = lax.dot_general(
        attn_weights, value,
        (contract_dims, ((), ())))
    return y

class DotProductAttention(nn.Module):
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  train: bool = False

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, bias=None, dtype=jnp.float32):
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    out_features = self.out_features or inputs_q.shape[-1]

    QKVDense = functools.partial(
      nn.Dense, features=qkv_features, use_bias=False, dtype=dtype)
    query = QKVDense(name='query')(inputs_q)
    key = QKVDense(name='key')(inputs_kv)
    value = QKVDense(name='value')(inputs_kv)

    y = RawDotProductAttention(train=self.train)(
        query, key, value, bias=bias, dtype=dtype)

    y = nn.Dense(features=out_features, dtype=dtype, name='out')(y)
    return y

class MultiHeadDotProductAttention(nn.Module):
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  batch_axes: Sequence[int] = (0,)
  num_heads: int = 1
  broadcast_dropout: bool = False
  train: bool = False
  @nn.compact
  def __call__(self, inputs_q, inputs_kv, bias=None, dtype=jnp.float32):
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    out_features = self.out_features or inputs_q.shape[-1]

    # Tekli Dikkat boyutundan Çoklu Dikkat (Multi-Headed) boyutuna geçiş.
    Attn = nn.vmap(DotProductAttention,
                   in_axes=(None, None, None),
                   out_axes=2,
                   axis_size=self.num_heads,
                   variable_axes={'params': 0},
                   split_rngs={'params': True,
                               'dropout': not self.broadcast_dropout})

    # Vmap batch boyutları arasında.
    for axis in reversed(sorted(self.batch_axes)):
      Attn = nn.vmap(Attn,
                     in_axes=(axis, axis, axis),
                     out_axes=axis,
                     variable_axes={'params': None},
                     split_rngs={'params': False, 'dropout': False})

    # vmap ile sınıflandırılmış girdiler.
    y = Attn(qkv_features=qkv_features // self.num_heads,
             out_features=out_features,
             train=self.train,
             name='attention')(inputs_q, inputs_kv, bias)

    return y.mean(axis=-2)


key1, key2, key3, key4 = random.split(random.PRNGKey(0), 4)
x = random.uniform(key1, (3, 13, 64))

model = functools.partial(
  MultiHeadDotProductAttention,
  broadcast_dropout=False,
  num_heads=2,
  batch_axes=(0,))

init_variables = model(train=False).init({'params': key2}, x, x)
print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))

y = model(train=True).apply(init_variables, x, x, rngs={'dropout': key4})
print('output:\n', y.shape)
