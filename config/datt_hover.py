import numpy as np

from config.configuration import *

drone_config = DroneConfiguration(
    mass=ConfigValue[float](default=1.0, randomize=False),
    I=ConfigValue[float](default=1.0, randomize=False),
    # g = ConfigValue[float](9.8, False)
)

wind_config = WindConfiguration(
    is_wind=False,
    dir=ConfigValue[np.ndarray](default=np.zeros(3), randomize=False),
)

init_config = InitializationConfiguration()

sim_config = SimConfiguration()

adapt_config = AdaptationConfiguration()

train_config = TrainingConfiguration()

policy_config = PolicyConfiguration(fb_term=False, ff_term=False)

ref_config = RefConfiguration()

config = AllConfig(
    drone_config=drone_config,
    wind_config=wind_config,
    init_config=init_config,
    sim_config=sim_config,
    adapt_config=adapt_config,
    training_config=train_config,
    policy_config=policy_config,
    ref_config=ref_config,
)
