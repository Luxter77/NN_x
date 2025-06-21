from torch import Tensor, pi

def linear_ramp_up(t: Tensor, t_max: Tensor) -> Tensor:
    """Linear ramp-up from 0 to 1 over [0, t_max]."""
    return (t / t_max).clamp(max=1.0)

def exponential_ramp_up(t: Tensor, t_max: Tensor) -> Tensor:
    """Exponential ramp-up with slow start and rapid rise near t_max."""
    return ( -5 * (1 - t / t_max).pow(2) ).exp()

def sigmoid_ramp_up(t: Tensor, k: Tensor, t_mid: Tensor) -> Tensor:
    """Sigmoid ramp-up centered at t_mid, sharpness controlled by k."""
    return 1 / ( 1 + (-k * (t - t_mid)).exp() )

def cosine_ramp_up(t: Tensor, t_max: Tensor) -> Tensor:
    """Cosine ramp-up from 0 to 1 over [0, t_max]."""
    return 0.5 * (1 - (pi * t / t_max).cos())

def polynomial_ramp_up(t: Tensor, t_max: Tensor, pow: Tensor) -> Tensor:
    """Polynomial ramp-up with controllable exponent."""
    return (t / t_max).pow(pow)

def exponential_decay(t: Tensor, k: Tensor, start: Tensor) -> Tensor:
    """Exponential decay from start, with decay rate k."""
    return start * (-k * t).exp()

def step_decay(t: Tensor, step: Tensor, start: Tensor, gamma: Tensor):
    """Stepwise decay by factor gamma every `step` steps."""
    return start * gamma.pow(t // step)

def linear_decay(t: Tensor, t_max: Tensor) -> Tensor:
    """Linear decay from 1 to 0 over [0, t_max]."""
    return (1 - t / t_max).clamp(min=0.0)

def cosine_decay(t: Tensor, t_max: Tensor) -> Tensor:
    """Cosine decay from 1 to 0 over [0, t_max]."""
    return 0.5 * ( 1 + (pi * t / t_max).cos() )

def inverse_time_decay(t: Tensor, k: Tensor, start: Tensor) -> Tensor:
    """Inverse time decay: slow decrease over time controlled by k."""
    return start / (1 + k * t)

# For restarts: caller must handle resets with t % T.
def cosine_warmup(t: Tensor, T: Tensor, eta_min: Tensor, eta_max: Tensor, ):
    """Cosine warm-up from eta_min to eta_max."""
    return eta_min + 0.5 * ( eta_max - eta_min ) * ( 1 - (pi * t / T).cos() )

# For restarts: caller must handle resets with t % T.
def cosine_annealing(t: Tensor, T: Tensor, eta_min: Tensor, eta_max: Tensor, ):
    """Cosine annealing from eta_max to eta_min."""
    return eta_min + 0.5 * ( eta_max - eta_min ) * ( 1 + (pi * t / T).cos() )

def gaussian_bump(t: Tensor, center: Tensor, width: Tensor) -> Tensor:
    """Gaussian-shaped bump centered at `center` with std `width`."""
    return ( -(t - center).pow(2) / (2 * width.pow(2))).exp()

def polynomial_decay(t: Tensor, t_max: Tensor, pow: Tensor) -> Tensor:
    """Polynomial decay from 1 to 0 with controllable exponent."""
    return (1 - t / t_max).pow(pow)

def logaritmic_approach(t: Tensor, target: Tensor, rate: Tensor) -> Tensor:
    """Logarithmic increase from 0 to 1, approaching slowly near target."""
    return (rate * t).log1p() / (rate * target).log1p()
