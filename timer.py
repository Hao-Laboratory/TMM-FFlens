"""
timing tools
"""

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Timer
^^^^^
"""
import logging
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl

# from pytorch_lightning.trainer.states import RunningStage
# from pytorch_lightning.utilities import LightningEnum

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_info

log = logging.getLogger(__name__)


class Timer(pl.Callback):
    """The Timer callback tracks the time spent in the training, validation, and test loops and interrupts the
    Trainer if the given time limit for the training loop is reached.

    Args:
        duration: A string in the format DD:HH:MM:SS (days, hours, minutes seconds), or a :class:`datetime.timedelta`,
            or a dict containing key-value compatible with :class:`~datetime.timedelta`.
        interval: Determines if the interruption happens on epoch level or mid-epoch.
            Can be either ``"epoch"`` or ``"step"``.
        verbose: Set this to ``False`` to suppress logging messages.

    Raises:
        MisconfigurationException:
            If ``interval`` is not one of the supported choices.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Timer

        # stop training after 12 hours
        timer = Timer(duration="00:12:00:00")

        # or provide a datetime.timedelta
        from datetime import timedelta
        timer = Timer(duration=timedelta(weeks=1))

        # or provide a dictionary
        timer = Timer(duration=dict(weeks=4, days=2))

        # force training to stop after given time limit
        trainer = Trainer(callbacks=[timer])

        # query training/validation/test time (in seconds)
        timer.time_elapsed("train")
        timer.start_time("validate")
        timer.end_time("test")
    """

    def __init__(
            self,
            duration: Optional[Union[str, timedelta, Dict[str, int]]] = None,
            verbose: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(duration, str):
            dhms = duration.strip().split(":")
            dhms = [int(i) for i in dhms]
            duration = timedelta(days=dhms[0], hours=dhms[1], minutes=dhms[2], seconds=dhms[3])
        if isinstance(duration, dict):
            duration = timedelta(**duration)

        self._duration = duration.total_seconds() if duration is not None else None

        self._verbose = verbose
        self._start_time = {'train': None}
        self._end_time = {'train': None}
        self._offset = 0

    def start_time(self, ) -> Optional[float]:
        """Return the start time of a particular 'train' (in seconds)"""

        return self._start_time['train']

    def end_time(self, ) -> Optional[float]:
        """Return the end time of a particular 'train' (in seconds)"""

        return self._end_time['train']

    def time_elapsed(self, ) -> float:
        """Return the time elapsed for a particular 'train' (in seconds)"""
        start = self.start_time()
        end = self.end_time()
        offset = self._offset
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
        return end - start + offset

    def time_remaining(self, ) -> Optional[float]:
        """Return the time remaining for a particular 'train' (in seconds)"""
        if self._duration is not None:
            return self._duration - self.time_elapsed()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._start_time['train'] = time.monotonic()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._end_time['train'] = time.monotonic()

    def on_fit_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        # this checks the time after the state is reloaded, regardless of the interval.
        # this is necessary in case we load a state whose timer is already depleted
        if self._duration is None:
            return
        self._check_time_remaining(trainer)

    def on_train_batch_end(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:

        self._check_time_remaining(trainer)

    def state_dict(self) -> Dict[str, Any]:
        return {"time_elapsed": {'train': self.time_elapsed()}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        time_elapsed = state_dict.get("time_elapsed", {})
        self._offset = time_elapsed.get('train', 0)

    def _check_time_remaining(self, trainer: "pl.Trainer") -> None:
        assert self._duration is not None
        should_stop = self.time_elapsed() >= self._duration
        # should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and self._verbose:
            elapsed = timedelta(seconds=int(self.time_elapsed()))
            rank_zero_info(f"Time limit reached. Elapsed time is {elapsed}. Signaling Trainer to stop.")
