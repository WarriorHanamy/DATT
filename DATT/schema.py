"""Shared Pydantic schema for INPUT config, OUTPUT metadata, and POLICY semantics.

Layers
------
1. ConfigValue + sub-config models  -- INPUT: what the simulator/policy receives
2. ObsGroup / ActionGroup            -- I/O contract: named slices of flat vectors
3. PolicySpec                        -- POLICY SEMANTICS: complete contract of a policy
4. PolicyManifest                    -- OUTPUT: sidecar .json accompanying a checkpoint
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


# ============================================================================
# EnvCondition -- standard Enum, used across the codebase
# ============================================================================


class EnvCondition(Enum):
    WIND = "wind"
    MASS = "mass"
    I = "moment_of_inertia"  # noqa: E741
    LATENCY = "latency"
    K = "k"
    KW = "kw"
    KT = "kt"

    def get_attribute(self, env):
        """Given the env, return (attr, dim, min, max) for this condition."""
        return {
            EnvCondition.WIND: (env.wind_vector, 3, -2.0, 2.0),
            EnvCondition.MASS: (env.model.mass, 1, 0.0, np.inf),
            EnvCondition.I: (np.diagonal(env.model.I), 3, 0.0, np.inf),
            EnvCondition.LATENCY: (env.latency, 1, 0.0, np.inf),
            EnvCondition.K: (env.k, 1, 0.0, 1.0),
            EnvCondition.KW: (env.kw, 1, 0.0, 1.0),
            EnvCondition.KT: (env.kt, 1, 0.0, 1.0),
        }[EnvCondition(self._value_)]


# ============================================================================
# ConfigValue -- randomizable parameter
# ============================================================================


class ConfigValue(BaseModel, Generic[T]):
    """
    A parameter that can be randomized, with bounds.

    ``T`` is the semantic type of the parameter (float, int, np.ndarray, bool).
    The model_config ``arbitrary_types_allowed`` permits numpy array values.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default: Any = Field(..., description="Default value when not randomized")
    randomize: bool = Field(False, description="Enable random sampling")
    min: Any = Field(None, description="Lower bound [unit of T]")
    max: Any = Field(None, description="Upper bound [unit of T]")

    @model_validator(mode="after")
    def _check_min_max(self):
        if self.randomize and (self.min is None or self.max is None):
            raise ValueError("Must specify min and max when randomize=True")
        return self

    def get_value(self) -> Any:
        """Return the parameter value, randomized if configured."""
        if not self.randomize:
            return self.default
        if isinstance(self.default, np.ndarray):
            return np.random.uniform(self.min, self.max, self.default.shape)  # type: ignore[arg-type]
        return np.random.uniform(self.min, self.max)

    def __call__(self) -> Any:
        return self.get_value()


# ============================================================================
# Sampler -- plain class for parameter sampling
# ============================================================================


class Sampler:
    """Sampling strategy for groups of ConfigValue parameters."""

    def __init__(self, sampling_func=None, name="custom_sampler"):
        if sampling_func is None:
            self.sampling_func = self.default_sample
            self.name = "uniform"
        else:
            self.sampling_func = sampling_func
            self.name = name

    def sample_param(self, param: ConfigValue, **kwargs):
        if param.randomize:
            return self.sampling_func(param, **kwargs)
        return param.default

    @staticmethod
    def default_sample(param: ConfigValue, **kwargs):
        return np.random.uniform(param.min, param.max)


# ============================================================================
# Sub-configuration models
# ============================================================================


class DroneConfiguration(BaseModel):
    """Physical drone parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mass: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=1.0, randomize=False),
        description="Drone mass [kg]",
    )
    I: ConfigValue = Field(  # noqa: E741
        default_factory=lambda: ConfigValue(default=1.0, randomize=False),
        description="Moment of inertia (diagonal, all axes equal) [kg·m²]",
    )
    sampler: Sampler = Field(default_factory=Sampler, exclude=True)


class WindConfiguration(BaseModel):
    """Wind disturbance parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_wind: bool = Field(False, description="Enable wind disturbance")
    dir: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=np.zeros(3), randomize=False),
        description="Wind direction vector [m/s]",
    )
    random_walk: bool = Field(False, description="Enable brownian motion on wind")
    sampler: Sampler = Field(default_factory=Sampler, exclude=True)


class InitializationConfiguration(BaseModel):
    """Initial state randomization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pos: ConfigValue = Field(
        default_factory=lambda: ConfigValue(
            default=np.array([0.0, 0.0, 0.0]),
            randomize=True,
            min=np.array([-0.5, -0.5, -0.5]),
            max=np.array([0.5, 0.5, 0.5]),
        ),
        description="Initial position [m]",
    )
    vel: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=np.zeros(3), randomize=False),
        description="Initial velocity [m/s]",
    )
    rot: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=np.zeros(3), randomize=False),
        description="Initial Euler ZYX rotation [deg]",
    )
    ang: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=np.zeros(3), randomize=False),
        description="Initial angular velocity [rad/s]",
    )
    sampler: Sampler = Field(default_factory=Sampler, exclude=True)


class SimConfiguration(BaseModel):
    """Simulator settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    linear_var: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0.0, randomize=False),
        description="Linear dynamics variance",
    )
    angular_var: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0.0, randomize=False),
        description="Angular dynamics variance",
    )
    obs_noise: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0.0, randomize=False),
        description="Observation noise std [m]",
    )
    latency: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0, randomize=False),
        description="Motor latency [multiples of dt]",
    )
    k: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=1.0, randomize=False),
        description="First-order motor delay coefficient [-]",
    )
    kw: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0.4, randomize=False),
        description="Second-order delay omega coefficient [-]",
    )
    kt: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0.6, randomize=False),
        description="Second-order delay thrust coefficient [-]",
    )
    dt: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=0.02, randomize=False),
        description="Simulation timestep [s]",
    )
    g: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=9.8, randomize=False),
        description="Gravity [m/s²]",
    )
    second_order_delay: ConfigValue = Field(
        default_factory=lambda: ConfigValue(default=False, randomize=False),
        description="Use second-order delay model (vs first-order)",
    )
    L1_simulation: bool = Field(False, description="Run L1 adaptation in sim")
    sampler: Sampler = Field(default_factory=Sampler, exclude=True)


class TrainingConfiguration(BaseModel):
    """Training hyperparameters."""

    body_frame: bool = Field(True, description="Observations in body frame")
    env_diff_seed: bool = Field(False, description="Different seed per parallel env")
    reset_freq: int = Field(0, description="Ref trajectory change frequency [episodes]")
    reset_thresh: int = Field(5000, description="Episodes before ref randomization begins")


class PolicyConfiguration(BaseModel):
    """Neural network policy structure."""

    time_horizon: int = Field(10, description="Feedforward trajectory horizon [steps]")
    fb_term: bool = Field(True, description="Include feedback state term")
    ff_term: bool = Field(True, description="Include feedforward trajectory term")
    conv_extractor: bool = Field(True, description="Use conv feature extractor")


class AdaptationConfiguration(BaseModel):
    """Adaptation module settings."""

    include: list[EnvCondition] = Field(
        default_factory=list,
        description="Env parameters included in observation",
    )
    time_horizon: int = Field(50, description="Adaptation history length [steps]")

    def get_e_dim(self) -> int:
        e_dim = 0
        for cond in self.include:
            if cond == EnvCondition.WIND:
                e_dim += 3
            elif cond == EnvCondition.MASS:
                e_dim += 1
            elif cond == EnvCondition.I:
                e_dim += 3
            elif cond == EnvCondition.LATENCY:
                e_dim += 1
            elif cond == EnvCondition.K:
                e_dim += 1
            elif cond == EnvCondition.KW:
                e_dim += 1
            elif cond == EnvCondition.KT:
                e_dim += 1
        return e_dim


class RefConfiguration(BaseModel):
    """Reference trajectory parameters."""

    y_max: float = Field(1.0, description="Max y deviation [m]")
    z_max: float = Field(0.0, description="Max z deviation [m]")
    env_diff_seed: bool = Field(False, description="Different seed per env")
    diff_axis: bool = Field(False, description="Differ by axis")
    include_all: bool = Field(False, description="Include all ref types in mixed")
    seed: Optional[int] = Field(None, description="Random seed for ref generation")
    init_ref: Optional[int] = Field(None, description="Initial ref for mixed trajectory")
    ref_filename: str = Field("", description="Filename for gen_traj ref")
    default_ref: str = Field("random_zigzag", description="Default trajectory when not specified")


class AllConfig(BaseModel):
    """Top-level configuration aggregating all sub-configs."""

    drone_config: DroneConfiguration
    wind_config: WindConfiguration
    init_config: InitializationConfiguration
    sim_config: SimConfiguration
    adapt_config: Optional[AdaptationConfiguration] = None
    training_config: TrainingConfiguration = Field(default_factory=TrainingConfiguration)
    policy_config: PolicyConfiguration = Field(default_factory=PolicyConfiguration)
    ref_config: RefConfiguration = Field(default_factory=RefConfiguration)


# ============================================================================
# Observation / Action named groups  (the I/O contract)
# ============================================================================


class ObsGroup(BaseModel):
    """Named slice of the flattened observation vector."""

    name: str = Field(..., description="Short mnemonic: pos_bf, vel_bf, quat, wind, ...")
    dim: int = Field(..., description="Number of scalar channels")
    start: int = Field(..., description="First index in flattened observation array")
    low: float = Field(-50.0, description="Lower bound")
    high: float = Field(50.0, description="Upper bound")
    description: str = Field("", description="Physical meaning with unit [unit]")


class ActionGroup(BaseModel):
    """Named component of the action vector."""

    name: str = Field(..., description="Short mnemonic: thrust, angvel_x, angvel_y, angvel_z")
    dim: int = Field(1, description="Scalar dimension count")
    start: int = Field(..., description="First index in flattened action array")
    low: float = Field(-20.0, description="Lower bound")
    high: float = Field(20.0, description="Upper bound")
    description: str = Field("", description="Physical meaning with unit [unit]")


_EXTRA_COND_MAP: dict[EnvCondition, tuple] = {
    EnvCondition.WIND: ("wind", 3, -2.0, 2.0, "Wind disturbance vector [m/s]"),
    EnvCondition.MASS: ("mass", 1, 0.0, 100.0, "Mass perturbation [kg]"),
    EnvCondition.I: ("I", 3, 0.0, 100.0, "Inertia diagonal [kg·m²]"),
    EnvCondition.LATENCY: ("latency", 1, 0.0, 100.0, "Motor latency [steps]"),
    EnvCondition.K: ("k", 1, 0.0, 1.0, "First-order delay coefficient [-]"),
    EnvCondition.KW: ("kw", 1, 0.0, 1.0, "Second-order delay ω coeff [-]"),
    EnvCondition.KT: ("kt", 1, 0.0, 1.0, "Second-order delay thrust coeff [-]"),
}

_DEFAULT_ACTION_GROUPS: list[ActionGroup] = [
    ActionGroup(
        name="thrust", dim=1, start=0, low=-20, high=20, description="Collective thrust [m/s²]"
    ),
    ActionGroup(
        name="angvel_x",
        dim=1,
        start=1,
        low=-20,
        high=20,
        description="Body angular velocity x [rad/s]",
    ),
    ActionGroup(
        name="angvel_y",
        dim=1,
        start=2,
        low=-20,
        high=20,
        description="Body angular velocity y [rad/s]",
    ),
    ActionGroup(
        name="angvel_z",
        dim=1,
        start=3,
        low=-20,
        high=20,
        description="Body angular velocity z [rad/s]",
    ),
]


# ============================================================================
# PolicySpec -- the central POLICY SEMANTICS type
# ============================================================================


class PolicySpec(BaseModel):
    """
    Complete I/O contract of a DATT policy.

    Derived from AllConfig + task -- NOT stored as static data.
    The ``obs_groups`` / ``action_groups`` document the flattened vectors
    so every index has a named, documented physical meaning.
    """

    # -- identity --
    name: str = Field(..., description="Unique policy name")
    task: str = Field(..., description="DroneTask name (hover, trajectory_fbff, ...)")
    algo: str = Field(..., description="RL algorithm (ppo, sac, ...)")

    # -- observation contract --
    body_frame: bool = Field(True, description="Observations expressed in body frame?")
    obs_groups: list[ObsGroup] = Field(
        ..., description="Semantic grouping of flattened observation vector"
    )
    obs_dim: int = Field(..., description="Total observation vector length")

    # -- action contract --
    action_groups: list[ActionGroup] = Field(..., description="Semantic grouping of action vector")
    action_dim: int = Field(4, description="Total action vector length")
    action_bounds: tuple[float, float] = Field(
        (-20.0, 20.0), description="Action space bounds [rad/s, m/s²]"
    )

    # -- policy structure --
    time_horizon: int = Field(10, description="Feedforward trajectory horizon [steps]")
    fb_term: bool = Field(True, description="Include feedback state term")
    ff_term: bool = Field(True, description="Include feedforward trajectory term")
    net_arch: list[int] = Field(
        default_factory=lambda: [64, 64, 64],
        description="MLP hidden layer sizes [pi, vf share this]",
    )

    # -- adaptation --
    adaptive: bool = Field(False, description="Policy uses adaptation?")
    adaptation_type: Optional[str] = Field(None, description="l1 | naive | rma")
    adapt_network: Optional[str] = Field(None, description="RMA adapt net checkpoint name")
    env_conditions: list[str] = Field(
        default_factory=list, description="Env params exposed to policy observation"
    )

    # -- provenance (filled at load time, not stored independently) --
    checkpoint_path: str = Field("", description="Path to .zip file", exclude=True)
    config_file: str = Field("", description="Config .py filename used for training", exclude=True)

    @property
    def obs_names(self) -> list[str]:
        """Flat list of channel names for every index in the observation vector."""
        names: list[str] = []
        for g in self.obs_groups:
            if g.dim == 1:
                names.append(g.name)
            else:
                for i in range(g.dim):
                    names.append(f"{g.name}[{i}]")
        return names

    @classmethod
    def from_config(
        cls,
        name: str,
        task: str,
        algo: str,
        config: AllConfig,
        *,
        adaptive: bool = False,
        adaptation_type: Optional[str] = None,
        adapt_network: Optional[str] = None,
        checkpoint_path: str = "",
        config_file: str = "",
    ) -> "PolicySpec":
        """Derive the full I/O contract from an AllConfig + task."""
        body_frame = config.training_config.body_frame
        adapt_cfg = config.adapt_config
        included: list[EnvCondition] = adapt_cfg.include if adapt_cfg else []
        is_trajectory = task.startswith("trajectory")
        fb_term = config.policy_config.fb_term
        ff_term = config.policy_config.ff_term
        time_horizon = config.policy_config.time_horizon

        # --- build obs_groups in order ---
        obs_groups: list[ObsGroup] = []

        pos_label = "pos_bf" if body_frame else "pos_world"
        vel_label = "vel_bf" if body_frame else "vel_world"
        frame_tag = "body frame" if body_frame else "world frame"

        # Determine how position / feedback are arranged for trajectory tasks.
        # fb_term=True: pos is raw position, fb is appended after quat.
        # fb_term=False: pos is replaced by position error (fb).
        if is_trajectory and not fb_term:
            # Position replaced by position error
            obs_groups.append(
                ObsGroup(
                    name="pos_err_bf" if body_frame else "pos_err_world",
                    dim=3,
                    start=0,
                    low=-50,
                    high=50,
                    description=f"Position error (pos - ref) in {frame_tag} [m]",
                )
            )
        else:
            obs_groups.append(
                ObsGroup(
                    name=pos_label,
                    dim=3,
                    start=0,
                    low=-50,
                    high=50,
                    description=f"Position in {frame_tag} [m]",
                )
            )

        obs_groups.append(
            ObsGroup(
                name=vel_label,
                dim=3,
                start=3,
                low=-50,
                high=50,
                description=f"Velocity in {frame_tag} [m/s]",
            )
        )
        obs_groups.append(
            ObsGroup(
                name="quat",
                dim=4,
                start=6,
                low=-50,
                high=50,
                description="Attitude quaternion (w, x, y, z) [-]",
            )
        )

        start = 10

        if is_trajectory and fb_term:
            obs_groups.append(
                ObsGroup(
                    name="fb_err" if body_frame else "fb_err_world",
                    dim=3,
                    start=start,
                    low=-50,
                    high=50,
                    description="Current position error (pos - ref) [m]",
                )
            )
            start += 3

        # --- env conditions ---
        env_conditions: list[str] = []
        for cond in included:
            cname, cdim, clo, chi, cdesc = _EXTRA_COND_MAP[cond]
            obs_groups.append(
                ObsGroup(name=cname, dim=cdim, start=start, low=clo, high=chi, description=cdesc)
            )
            start += cdim
            env_conditions.append(cond.value)

        # --- feedforward waypoints (trajectory tasks with ff_term) ---
        if is_trajectory and ff_term:
            obs_groups.append(
                ObsGroup(
                    name="ff_waypoints",
                    dim=3 * time_horizon,
                    start=start,
                    low=-50,
                    high=50,
                    description="Future position errors at 3·dt·[1..H] lookahead [m]",
                )
            )
            start += 3 * time_horizon

        obs_dim = sum(g.dim for g in obs_groups)

        return cls(
            name=name,
            task=task,
            algo=algo,
            body_frame=body_frame,
            obs_groups=obs_groups,
            obs_dim=obs_dim,
            action_groups=list(_DEFAULT_ACTION_GROUPS),
            action_dim=4,
            time_horizon=time_horizon,
            fb_term=fb_term,
            ff_term=ff_term,
            adaptive=adaptive,
            adaptation_type=adaptation_type,
            adapt_network=adapt_network,
            env_conditions=env_conditions,
            checkpoint_path=checkpoint_path,
            config_file=config_file,
        )


# ============================================================================
# PolicyManifest -- sidecar .json for a checkpoint (OUTPUT layer)
# ============================================================================


class PolicyManifest(BaseModel):
    """
    Sidecar metadata file accompanying a checkpoint (``.policy.json``).

    Stored alongside ``{name}.zip``.  If missing for old checkpoints,
    ``PolicySpec`` is constructed from the config .py file instead.
    """

    policy: PolicySpec = Field(..., description="Policy I/O contract")
    total_steps: int = Field(..., description="Training steps completed")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp (UTC)",
    )


# ============================================================================
# PresetEntry -- typed preset for CLI dispatch
# ============================================================================


class TrainingPreset(BaseModel):
    """Preset mapping for ``datt train``."""

    config: str = Field(..., description="Config filename (e.g. datt.py)")
    task: str = Field(..., description="DroneTask value")
    algo: str = Field(..., description="RL algorithm (ppo, sac, ...)")


class EvalPreset(BaseModel):
    """Preset mapping for ``datt eval``."""

    checkpoint: str = Field(..., description="Checkpoint name (stem of .zip)")
    config: str = Field(..., description="Config filename (e.g. datt.py)")
    task: str = Field(..., description="DroneTask value")
    algo: str = Field(..., description="RL algorithm (ppo, sac, ...)")


class RMAEvalPreset(BaseModel):
    """Preset mapping for ``datt eval-rma``."""

    checkpoint: str = Field(..., description="Checkpoint name (stem of .zip)")
    adapt_name: str = Field(..., description="RMA adaptation network name")
    config: str = Field(..., description="Config filename")
    task: str = Field(..., description="DroneTask value")
    algo: str = Field(..., description="RL algorithm (ppo, sac, ...)")


class RMATrainingPreset(BaseModel):
    """Preset mapping for ``datt train-rma``."""

    config: str = Field(..., description="Config filename")
    task: str = Field(..., description="DroneTask value")
    algo: str = Field(..., description="RL algorithm (ppo, sac, ...)")


# ============================================================================
# Utility: load a PolicySpec from the best available source
# ============================================================================


def load_policy_spec(
    policy_name: str,
    task: str,
    algo: str,
    config: AllConfig,
    *,
    adaptive: bool = False,
    adaptation_type: Optional[str] = None,
    adapt_network: Optional[str] = None,
    config_file: str = "",
) -> PolicySpec:
    """
    Return a PolicySpec for the given policy.

    Priority order:
    1. Sidecar ``{policy_name}.policy.json`` (if it exists)
    2. Derived from ``config`` + ``task`` / ``algo`` (fallback for old checkpoints)
    """
    from pathlib import Path

    import DATT

    _cp_dir = Path(DATT.__file__).resolve().parent.parent / "checkpoints"
    _cp_zip = _cp_dir / f"{policy_name}.zip"
    sidecar = _cp_dir / f"{policy_name}.policy.json"
    if sidecar.exists():
        manifest = PolicyManifest.model_validate_json(sidecar.read_text())
        spec = manifest.policy
        spec.checkpoint_path = str(_cp_zip)
        return spec

    return PolicySpec.from_config(
        name=policy_name,
        task=task,
        algo=algo,
        config=config,
        adaptive=adaptive,
        adaptation_type=adaptation_type,
        adapt_network=adapt_network,
        checkpoint_path=str(_cp_zip),
        config_file=config_file,
    )


def save_policy_manifest(
    policy_name: str,
    task: str,
    algo: str,
    config: AllConfig,
    total_steps: int,
    *,
    adaptive: bool = False,
    adaptation_type: Optional[str] = None,
    adapt_network: Optional[str] = None,
    config_file: str = "",
) -> PolicyManifest:
    """Build and write a .policy.json sidecar next to the checkpoint .zip."""
    from pathlib import Path

    import DATT

    spec = PolicySpec.from_config(
        name=policy_name,
        task=task,
        algo=algo,
        config=config,
        adaptive=adaptive,
        adaptation_type=adaptation_type,
        adapt_network=adapt_network,
        config_file=config_file,
    )
    manifest = PolicyManifest(policy=spec, total_steps=total_steps)

    sidecar = (
        Path(DATT.__file__).resolve().parent.parent / "checkpoints" / f"{policy_name}.policy.json"
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(manifest.model_dump_json(indent=2))
    return manifest
