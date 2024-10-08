
# ********************
# Time step schedulers
# ********************


class TimeStepScheduler(Protocol):

  def __call__(self, scheme: diffusion.Diffusion, *args, **kwargs) -> Array:
    """Outputs the time steps based on diffusion noise schedule."""
    ...


def uniform_time(
    scheme: diffusion.Diffusion,
    num_steps: int = 256,
    end_time: float | None = 1e-3,
    end_sigma: float | None = None,
) -> Array:
  """Time steps uniform in [t_min, t_max]."""
  if (end_time is None and end_sigma is None) or (
      end_time is not None and end_sigma is not None
  ):
    raise ValueError(
        "Exactly one of `end_time` and `end_sigma` must be specified."
    )

  start = diffusion.MAX_DIFFUSION_TIME
  end = end_time or scheme.sigma.inverse(end_sigma)
  return jnp.linspace(start, end, num_steps)


def exponential_noise_decay(
    scheme: diffusion.Diffusion,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3,
) -> Array:
  """Time steps corresponding to exponentially decaying sigma."""
  exponent = jnp.arange(num_steps) / (num_steps - 1)
  r = end_sigma / scheme.sigma_max
  sigma_schedule = scheme.sigma_max * jnp.power(r, exponent)
  return jnp.asarray(scheme.sigma.inverse(sigma_schedule))


def edm_noise_decay(
    scheme: diffusion.Diffusion,
    rho: int = 7,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3,
) -> Array:
  """Time steps corresponding to Eq. 5 in Karras et al."""
  rho_inv = 1 / rho
  sigma_schedule = jnp.arange(num_steps) / (num_steps - 1)
  sigma_schedule *= jnp.power(end_sigma, rho_inv) - jnp.power(
      scheme.sigma_max, rho_inv
  )
  sigma_schedule += jnp.power(scheme.sigma_max, rho_inv)
  sigma_schedule = jnp.power(sigma_schedule, rho)
  return jnp.asarray(scheme.sigma.inverse(sigma_schedule))