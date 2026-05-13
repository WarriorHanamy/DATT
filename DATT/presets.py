"""Preset mappings: semantic labels -> config, task, checkpoint combos."""

from DATT.schema import EvalPreset, RMAEvalPreset, RMATrainingPreset, TrainingPreset

TRAIN_PRESETS = {
    "hover": TrainingPreset(
        config="datt_hover.py",
        task="hover",
        algo="ppo",
    ),
    "tracking": TrainingPreset(
        config="datt.py",
        task="trajectory_fbff",
        algo="ppo",
    ),
    "wind": TrainingPreset(
        config="datt_wind_adaptive.py",
        task="trajectory_fbff",
        algo="ppo",
    ),
}

EVAL_PRESETS = {
    "hover": EvalPreset(
        checkpoint="datt_hover",
        config="datt_hover.py",
        task="hover",
        algo="ppo",
    ),
    "tracking": EvalPreset(
        checkpoint="datt",
        config="datt.py",
        task="trajectory_fbff",
        algo="ppo",
    ),
    "wind": EvalPreset(
        checkpoint="datt_wind_adaptive",
        config="datt_wind_adaptive.py",
        task="trajectory_fbff",
        algo="ppo",
    ),
}

RMA_EVAL_PRESETS = {
    "wind": RMAEvalPreset(
        checkpoint="datt_wind_adaptive",
        adapt_name="wind_RMA",
        config="datt_wind_adaptive.py",
        task="trajectory_fbff",
        algo="ppo",
    ),
}

RMA_TRAIN_PRESETS = {
    "wind": RMATrainingPreset(
        config="datt_wind_adaptive.py",
        task="trajectory_fbff",
        algo="ppo",
    ),
}

TRAJECTORY_CHOICES = [
    "hover",
    "line",
    "circle",
    "zigzag",
    "square",
    "star",
    "polygon",
    "poly",
    "chained_poly",
    "setpoint",
    "mixed",
    "zigzag_yaw",
]

TRAJECTORY_REF_MAP = {
    "hover": "hover",
    "line": "line_ref",
    "circle": "circle_ref",
    "zigzag": "random_zigzag",
    "square": "square_ref",
    "star": "pointed_star",
    "polygon": "closed_poly",
    "poly": "poly_ref",
    "chained_poly": "chained_poly_ref",
    "setpoint": "setpoint",
    "mixed": "mixed_ref",
    "zigzag_yaw": "random_zigzag_yaw",
}
