"""
Monitoring infrastructure
"""
import os
import abc
import torch
import h5py

from . import timer


class BasicHook(metaclass=abc.ABCMeta):
    """
    Abstract periodic monitor
    """

    def __init__(self, interval=1, atend=True):
        super().__init__()
        self._interval = interval
        self._atend = atend
        # Last processed step (to avoid duplicate runs at end)
        self._last = None

    def _isvalid(self, step):
        return step == 1 or step % self._interval == 0

    def _islast(self, step):
        return self._last and self._last == step

    def _run(self, step, polygraph):
        raise NotImplementedError

    def mayberun(self, step, polygraph):
        """
        Monitors progress at given simulation step.
        """
        if not self._isvalid(step):
            return
        self._last = step
        self._run(step, polygraph)

    def conclude(self, step, polygraph):
        """
        Concludes monitoring.
        """
        if not self._atend or self._islast(step):
            return
        self._run(step, polygraph)


class MonitorHook(BasicHook):
    """
    Periodic monitor for performance measurements
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clock = timer.Timer()

    def _run(self, step, polygraph):
        if not self._clock.isrunning():
            assert step == 1
            self._clock.start()
            throughput = 0.0
        else:
            dt = self._clock.lap()
            throughput = ((step - 1) / dt) / 1000.0

        beliefs = polygraph.ndata["beliefs"]
        a, b = torch.sum(
            torch.le(beliefs, 0.5)
        ), torch.sum(torch.gt(beliefs, 0.5))
        msg = "[MON]"
        msg = f"{msg} step {step:04d}"
        msg = f"{msg} Ksteps/s {throughput:6.2f}"
        msg = f"{msg} A/B {a / (a + b):4.2f}/{b / (a + b):4.2f}"
        print(msg)


class SnapshotHook(BasicHook):
    """
    Periodic logger for agent beliefs (and reliability if present).
    Also saves reliability so the simulation can be resumed correctly.
    """

    def __init__(self, messages=False, location=None, filename="data.hd5", **kwargs):
        super().__init__(**kwargs)
        assert location and os.path.isdir(location)
        self._filename = os.path.join(location, filename)
        self._messages = messages

    def _run(self, step, polygraph):
        # Open HDF5 file in append mode
        f = h5py.File(self._filename, "a")

        # Store beliefs
        beliefs = polygraph.ndata["beliefs"].cpu().numpy()
        grp = f.require_group("beliefs")
        # Overwrite if step already exists (can happen on resume)
        if str(step) in grp:
            del grp[str(step)]
        grp.create_dataset(str(step), data=beliefs)

        # Store reliability if present
        # This is essential for resuming simulations that use UnreliableOp,
        # where reliability is randomly assigned once per repeat.
        if "reliability" in polygraph.ndata:
            reliability = polygraph.ndata["reliability"].cpu().numpy()
            grp_rel = f.require_group("reliability")
            # We only need to store it once — use step as key for consistency
            # but always overwrite so we always have the latest
            if str(step) in grp_rel:
                del grp_rel[str(step)]
            grp_rel.create_dataset(str(step), data=reliability)

        # Store payoffs/messages if requested
        if self._messages:
            payoffs = polygraph.ndata["payoffs"].cpu().numpy()
            grp = f.require_group("payoffs")
            if str(step) in grp:
                del grp[str(step)]
            grp.create_dataset(str(step), data=payoffs)

        f.close()
