class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  num_channels: tuple[int, ...]
  num_res_blocks: tuple[int, ...]
  downsample_ratio: tuple[int, ...]
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_attention: bool = False
  num_heads: int = 8
  channels_per_head: int = -1
  use_position_encoding: bool = False
  normalize_qk: bool = False
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, emb: Array, *, is_training: bool) -> list[Array]:
    assert (
        len(self.num_channels)
        == len(self.num_res_blocks)
        == len(self.downsample_ratio)
    )
    kernel_dim = x.ndim - 2
    res = np.asarray(x.shape[1:-1])
    skips = []
    h = layers.ConvLayer(
        features=128,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=default_init(1.0),
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="conv_in",
    )(x)
    skips.append(h)

    for level, channel in enumerate(self.num_channels):
      h = layers.DownsampleConv(
          features=channel,
          ratios=(self.downsample_ratio[level],) * kernel_dim,
          kernel_init=default_init(1.0),
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"res{'x'.join(res.astype(str))}.downsample_conv",
      )(h)
      res = res // self.downsample_ratio[level]
      for block_id in range(self.num_res_blocks[level]):
        h = ConvBlock(
            out_channels=channel,
            kernel_size=kernel_dim * (3,),
            padding=self.padding,
            dropout=self.dropout_rate,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"res{'x'.join(res.astype(str))}.down.block{block_id}",
        )(h, emb, is_training=is_training)

        if self.use_attention and level == len(self.num_channels) - 1:
          b, *hw, c = h.shape
          # Adding positional encoding, only in Dstack.
          if self.use_position_encoding:
            h = position_embedding(
                ndim=kernel_dim,
                name=f"res{'x'.join(res.astype(str))}.down.block.posenc{block_id}",
            )(h)
          h = AttentionBlock(
              num_heads=self.num_heads,
              precision=self.precision,
              dtype=self.dtype,
              normalize_qk=self.normalize_qk,
              param_dtype=self.param_dtype,
              name=f"res{'x'.join(res.astype(str))}.down.block{block_id}.attn",
          )(h.reshape(b, -1, c), is_training=is_training)
          h = ResConv1x(
              hidden_layer_size=channel * 2,
              out_channels=channel,
              precision=self.precision,
              dtype=self.dtype,
              param_dtype=self.param_dtype,
              name=f"res{'x'.join(res.astype(str))}.down.block{block_id}.res_conv_1x",
          )(h).reshape(b, *hw, c)
        skips.append(h)

    return skips
