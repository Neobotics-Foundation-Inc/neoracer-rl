"""
Classical controllers to measure a learned policy against.

A learning curve on its own says little. These are the drivers a tabular agent
has to beat, written so they take the same observation NeoRacerEnv gives and
return the same action it takes, which makes the comparison direct.

    from baselines import PDCentering, FollowTheGap, PurePursuit
    driver = PDCentering()
    throttle, steer = driver.act(obs)

Three of them read only the observation, which is what the learner sees:

    PDCentering    steer to keep the left and right LiDAR beams equal
    WallFollow     hold a fixed distance from one wall
    FollowTheGap   steer toward the longest free direction

PurePursuit instead follows the track's centre line and needs the car's true
pose, which the learner never gets. It is the strong reference, not a fair
opponent. Every controller says so in is_privileged.

Each controller has a frozen params record with its gains. run.py tunes them
by sweeping those records.

Not here yet: MPC, which needs a model of the car and a solver in the loop,
and SLAM, which solves a problem this simulator does not pose because the
track is known and the pose is exact.

Frame: +X forward, +Y left, +Z up. Steering is + for left, as in the contract.
"""

from dataclasses import dataclass, field, replace

import numpy as np

# The 8 LiDAR beams of the observation, every 45 degrees, in the order the
# environment packs them: index 0 is dead ahead and the angles go anticlockwise.
BEAM_ANGLES_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
AHEAD, FRONT_LEFT, LEFT, REAR_LEFT, BEHIND, REAR_RIGHT, RIGHT, FRONT_RIGHT = range(8)

# A beam that hits nothing reads -1. Treat it as this far, in metres, rather
# than as a wall just behind the car.
FAR_M = 10.0


def beams(obs: np.ndarray) -> np.ndarray:
    """The 8 LiDAR ranges in metres, with the no-hit reading replaced by FAR_M."""
    r = np.asarray(obs[:8], dtype=float).copy()
    r[r < 0.0] = FAR_M
    return r


def speed(obs: np.ndarray) -> float:
    """The car's own speed estimate, in m/s, from its wheel encoders."""
    return float(obs[19])


@dataclass(frozen=True)
class PDGains:
    """One proportional and derivative pair, with the derivative filtered.

    kp multiplies the error, kd its rate of change per second. tau is the time
    constant of the low pass filter on that rate, in seconds: the LiDAR is
    noisy and an unfiltered derivative mostly amplifies it.
    """

    kp: float
    kd: float = 0.0
    tau: float = 0.08


class _PD:
    """A PD term with a filtered derivative. Call reset between episodes."""

    def __init__(self, gains: PDGains) -> None:
        self.gains = gains
        self.reset()

    def reset(self) -> None:
        self._last_error = 0.0
        self._rate = 0.0
        self._started = False

    def __call__(self, error: float, dt: float) -> float:
        if not self._started:
            self._last_error, self._started = error, True
        raw = (error - self._last_error) / max(dt, 1e-6)
        alpha = dt / max(self.gains.tau + dt, 1e-6)
        self._rate += alpha * (raw - self._rate)
        self._last_error = error
        return self.gains.kp * error + self.gains.kd * self._rate


def _throttle_for(target_ms: float, obs: np.ndarray, gain: float) -> float:
    """A proportional speed hold. The car reaches about 3.3 m/s at full
    throttle, so a gain near 1 per m/s of error is a reasonable start."""
    return float(np.clip(gain * (target_ms - speed(obs)), 0.0, 1.0))


@dataclass(frozen=True)
class CenteringParams:
    # Swept by run.py tune.
    steer: PDGains = field(default_factory=lambda: PDGains(kp=2.2, kd=0.15))
    target_ms: float = 2.2  # m/s
    throttle_gain: float = 1.0  # throttle per m/s of speed error
    corner_m: float = 1.1  # slow down when the way ahead is shorter than this
    corner_slowdown: float = 0.55  # fraction of target_ms taken in a corner


class PDCentering:
    """Keep the car midway between the two walls.

    The error is the left beam minus the right beam, in metres, so it is zero
    in the middle of the corridor and positive when there is more room on the
    left. A PD on that error steers back. The car also slows when the beam
    straight ahead gets short, which is the only warning it has of a corner.
    """

    is_privileged = False
    name = "pd_centering"

    def __init__(self, params: CenteringParams = CenteringParams()) -> None:
        self.params = params
        self._pd = _PD(params.steer)

    def reset(self) -> None:
        self._pd.reset()

    def act(self, obs: np.ndarray, dt: float = 1 / 30, pose=None) -> tuple[float, float]:
        r = beams(obs)
        error = r[LEFT] - r[RIGHT]
        steer = float(np.clip(self._pd(error, dt), -1.0, 1.0))
        target = self.params.target_ms
        if r[AHEAD] < self.params.corner_m:
            target *= self.params.corner_slowdown
        return _throttle_for(target, obs, self.params.throttle_gain), steer


@dataclass(frozen=True)
class WallFollowParams:
    # Swept by run.py tune: a stiffer gain than the centering controller needs,
    # because one beam carries the whole signal. The standoff is half the 1.0 m
    # corridor, so on this track following a wall and centering agree; on a
    # track with only one wall they would not.
    steer: PDGains = field(default_factory=lambda: PDGains(kp=3.0, kd=0.3))
    standoff_m: float = 0.52  # how far to hold off the followed wall
    follow_left: bool = False
    target_ms: float = 2.8
    throttle_gain: float = 1.0
    corner_m: float = 1.1
    corner_slowdown: float = 0.55
    skew_weight: float = 0.9  # how much the heading term counts against distance


class WallFollow:
    """Hold a fixed distance from one wall.

    Two signals, both from the simulator's own wall following controller. The
    distance error is the side beam minus the standoff, in metres. The skew is
    the rear side beam minus the front side beam, also in metres, and it is
    positive when the nose points toward the wall, which is the only heading
    estimate the LiDAR gives. Following one wall keeps working where the other
    one disappears, which a centering controller cannot do.
    """

    is_privileged = False
    name = "wall_follow"

    def __init__(self, params: WallFollowParams = WallFollowParams()) -> None:
        self.params = params
        self._pd = _PD(params.steer)

    def reset(self) -> None:
        self._pd.reset()

    def act(self, obs: np.ndarray, dt: float = 1 / 30, pose=None) -> tuple[float, float]:
        r = beams(obs)
        p = self.params
        # toward is the steer sign that turns into the followed wall. Steering
        # is positive to the left, so following the right wall means a positive
        # distance error, too far from it, has to steer right.
        if p.follow_left:
            side, front, rear, toward = r[LEFT], r[FRONT_LEFT], r[REAR_LEFT], 1.0
        else:
            side, front, rear, toward = r[RIGHT], r[FRONT_RIGHT], r[REAR_RIGHT], -1.0
        distance_error = side - p.standoff_m
        skew = rear - front
        error = distance_error - p.skew_weight * skew
        steer = float(np.clip(toward * self._pd(error, dt), -1.0, 1.0))
        target = p.target_ms
        if r[AHEAD] < p.corner_m:
            target *= p.corner_slowdown
        return _throttle_for(target, obs, p.throttle_gain), steer


@dataclass(frozen=True)
class GapParams:
    steer_gain: float = 0.85  # steer per radian of bearing to the best gap
    smoothing: float = 0.55  # how much of the new steer to take each step
    target_ms: float = 2.0  # swept by run.py tune
    throttle_gain: float = 1.0
    corner_m: float = 1.2
    corner_slowdown: float = 0.5


class FollowTheGap:
    """Steer toward the longest free direction.

    With only 8 beams this is the crudest of the three, but it needs no model
    of the track at all: take the five beams that point forward, weight each
    one by how far it reaches, and aim at the weighted mean bearing. The
    weighting rather than a plain maximum stops the steering jumping by
    45 degrees whenever two beams swap places.
    """

    is_privileged = False
    name = "follow_the_gap"

    def __init__(self, params: GapParams = GapParams()) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        self._steer = 0.0

    def act(self, obs: np.ndarray, dt: float = 1 / 30, pose=None) -> tuple[float, float]:
        r = beams(obs)
        forward = (FRONT_RIGHT, AHEAD, FRONT_LEFT, LEFT, RIGHT)
        bearings = np.radians([-45.0, 0.0, 45.0, 90.0, -90.0])
        reach = np.array([r[i] for i in forward])
        weight = np.maximum(reach, 0.0) ** 3  # strongly prefer the open direction
        aim = float(np.sum(weight * bearings) / max(np.sum(weight), 1e-6))
        target_steer = float(np.clip(self.params.steer_gain * aim, -1.0, 1.0))
        self._steer += self.params.smoothing * (target_steer - self._steer)
        target = self.params.target_ms
        if r[AHEAD] < self.params.corner_m:
            target *= self.params.corner_slowdown
        return _throttle_for(target, obs, self.params.throttle_gain), self._steer


@dataclass(frozen=True)
class PursuitParams:
    lookahead_m: float = 0.75  # how far down the centre line to aim
    lookahead_per_ms: float = 0.28  # extra lookahead per m/s of speed
    wheelbase_m: float = 0.288  # the NeoRacer's wheelbase
    max_steer_rad: float = 0.4  # the steering limit the action is a fraction of
    # Swept by run.py tune. Knowing the centre line, this one can carry far
    # more speed than the LiDAR controllers, which only see a corner once a
    # beam reaches it.
    target_ms: float = 3.0
    throttle_gain: float = 1.0
    curvature_slowdown: float = 1.4  # how hard to brake for a tight corner


class PurePursuit:
    """Follow the track's centre line with the classic geometric law.

    Pick the point on the centre line one lookahead ahead of the car, measure
    its bearing in the car's frame, and steer by the arc that reaches it:
    atan(2 L sin(alpha) / d), where L is the wheelbase and d the lookahead.
    The lookahead grows with speed, which is what keeps it stable.

    This reads the car's true pose and the track's geometry, neither of which
    the learner gets, so it is a reference rather than an opponent on equal
    terms.
    """

    is_privileged = True
    name = "pure_pursuit"

    def __init__(
        self, centre: np.ndarray, params: PursuitParams = PursuitParams()
    ) -> None:
        self.centre = np.asarray(centre, dtype=float)
        self.params = params
        step = np.linalg.norm(np.diff(self.centre, axis=0, append=self.centre[:1]), axis=1)
        self._spacing = float(np.mean(step))

    def reset(self) -> None:
        return None

    def act(
        self, obs: np.ndarray, dt: float = 1 / 30, pose: tuple[float, float, float] | None = None
    ) -> tuple[float, float]:
        if pose is None:
            raise ValueError("PurePursuit needs the car's pose: pass pose=(x, y, yaw)")
        x, y, yaw = pose
        p = self.params
        here = np.array([x, y])
        nearest = int(np.argmin(np.sum((self.centre - here) ** 2, axis=1)))

        lookahead = p.lookahead_m + p.lookahead_per_ms * speed(obs)
        ahead = max(1, int(round(lookahead / max(self._spacing, 1e-6))))
        target = self.centre[(nearest + ahead) % len(self.centre)]

        delta = target - here
        bearing = float(np.arctan2(delta[1], delta[0]) - yaw)
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        distance = float(np.hypot(*delta))
        curvature = 2.0 * np.sin(bearing) / max(distance, 1e-6)
        steer_rad = float(np.arctan(p.wheelbase_m * curvature))
        steer = float(np.clip(steer_rad / p.max_steer_rad, -1.0, 1.0))

        target_ms = p.target_ms / (1.0 + p.curvature_slowdown * abs(curvature))
        return _throttle_for(target_ms, obs, p.throttle_gain), steer


def tuned(controller, **gains):
    """A copy of a controller with some params replaced, for sweeps."""
    return type(controller)(replace(controller.params, **gains))
