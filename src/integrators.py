from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


State = np.ndarray
RHS = Callable[[State], State]


def rk4_step(rhs: RHS, y: State, h: float) -> State:
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * h * k1)
    k3 = rhs(y + 0.5 * h * k2)
    k4 = rhs(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass(frozen=True)
class RK45Result:
    y: State
    h_next: float
    error: float
    accepted: bool


def rk45_step(rhs: RHS, y: State, h: float, atol: float = 1.0e-8, rtol: float = 1.0e-6) -> RK45Result:
    """Dormand-Prince RK45 step."""

    k1 = rhs(y)
    k2 = rhs(y + h * (1.0 / 5.0) * k1)
    k3 = rhs(y + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2))
    k4 = rhs(y + h * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2 + (32.0 / 9.0) * k3))
    k5 = rhs(
        y
        + h
        * (
            (19372.0 / 6561.0) * k1
            - (25360.0 / 2187.0) * k2
            + (64448.0 / 6561.0) * k3
            - (212.0 / 729.0) * k4
        )
    )
    k6 = rhs(
        y
        + h
        * (
            (9017.0 / 3168.0) * k1
            - (355.0 / 33.0) * k2
            + (46732.0 / 5247.0) * k3
            + (49.0 / 176.0) * k4
            - (5103.0 / 18656.0) * k5
        )
    )
    y5 = y + h * (
        (35.0 / 384.0) * k1
        + (500.0 / 1113.0) * k3
        + (125.0 / 192.0) * k4
        - (2187.0 / 6784.0) * k5
        + (11.0 / 84.0) * k6
    )
    k7 = rhs(y5)
    y4 = y + h * (
        (5179.0 / 57600.0) * k1
        + (7571.0 / 16695.0) * k3
        + (393.0 / 640.0) * k4
        - (92097.0 / 339200.0) * k5
        + (187.0 / 2100.0) * k6
        + (1.0 / 40.0) * k7
    )
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
    err = float(np.max(np.abs((y5 - y4) / scale)))
    if err == 0.0:
        factor = 2.0
    else:
        factor = float(np.clip(0.9 * err ** (-0.2), 0.2, 5.0))
    return RK45Result(y=y5, h_next=h * factor, error=err, accepted=err <= 1.0)

