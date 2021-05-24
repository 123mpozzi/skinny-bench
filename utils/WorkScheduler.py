from typing import Callable, Any
import tensorflow as tf
from .models import Model


class WorkScheduler:
    def __init__(self) -> None:
        self.data = []

    def add_data(self, model: Model,
                 func: Callable[[Model, Any], None],
                 **kwargs) -> None:
        self.data.append((model, func, kwargs))

    def do_work(self) -> None:
        for model, func, kwargs in self.data:
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            func(model=model, **kwargs)