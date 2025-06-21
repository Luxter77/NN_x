from torch import Tensor, pi

def linear_ramp_up(t: Tensor, t_max: Tensor) -> Tensor:
    return (t / t_max).clamp(max=1.0)

def exponential_ramp_up(t: Tensor, t_max: Tensor) -> Tensor:
    return ( -5 * (1 - t / t_max).pow(2) ).exp()

def sigmoid_ramp_up(t: Tensor, k: Tensor, t_mid: Tensor) -> Tensor:
    return 1 / ( 1 + (-k * (t - t_mid)).exp() )

def cosine_ramp_up(t: Tensor, t_max: Tensor) -> Tensor:
    return 0.5 * (1 - (pi * t / t_max).cos())

def polynomial_ramp_up(t: Tensor, t_max: Tensor, pow: Tensor) -> Tensor:
    return (t / t_max).pow(pow)

def exponential_decay(t: Tensor, k: Tensor, start: Tensor) -> Tensor:
    return start * (-k * t).exp()

def step_decay(t: Tensor, step: Tensor, start: Tensor, gamma: Tensor):
    return start * gamma.pow(t // step)

def linear_decay(t: Tensor, t_max: Tensor) -> Tensor:
    return (1 - t / t_max).clamp(min=0.0)

def cosine_decay(t: Tensor, t_max: Tensor) -> Tensor:
    return 0.5 * ( 1 + (pi * t / t_max).cos() )

def inverse_time_decay(t: Tensor, k: Tensor, start: Tensor) -> Tensor:
    return start / (1 + k * t)

def cosine_warmup_with_cold_restart(t: Tensor, T: Tensor, eta_min: Tensor, eta_max: Tensor, ):
    return eta_min + 0.5 * ( eta_max - eta_min ) * ( 1 - (pi * t / T).cos() )

def cosine_annealing_with_warm_restart(t: Tensor, T: Tensor, eta_min: Tensor, eta_max: Tensor, ):
    return eta_min + 0.5 * ( eta_max - eta_min ) * ( 1 + (pi * t / T).cos() )

def gaussian_bump(t: Tensor, center: Tensor, width: Tensor) -> Tensor:
    return ( -(t - center).pow(2) / (2 * width.pow(2))).exp()

def polynomial_decay(t: Tensor, t_max: Tensor, pow: Tensor) -> Tensor:
    return (1 - t / t_max).pow(pow)

def logaritmic_approach(t: Tensor, target: Tensor, rate: Tensor) -> Tensor:
    return (rate * t).log1p() / (rate * target).log1p()
