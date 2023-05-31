from baseline import BaseTuner
from types import MappingProxyType

def make_default_tuner_type_mutable(tuner: BaseTuner) -> BaseTuner:
    for i in tuner.__dict__.keys():
        if isinstance(tuner.__dict__[i], MappingProxyType):
            setattr(tuner, i, dict(tuner.__dict__[i]))

    return tuner