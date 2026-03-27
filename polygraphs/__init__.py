"""
PolyGraph module
"""
import os
import uuid
import datetime
import random as rnd
import collections
import json
import pickle

import dgl
import torch
import numpy as np
import h5py

from . import hyperparameters as hparams
from . import metadata
from . import monitors
from . import graphs
from . import ops
from . import logger
from . import timer

log = logger.getlogger()

# Cache data directory for all results
_RESULTCACHE = os.getenv("POLYGRAPHS_CACHE") or "~/polygraphs-cache/results"

# Checkpoint filename (saved in results directory)
_CHECKPOINT_FILE = "checkpoint.pkl"


def _mkdir(directory="auto", attempts=10):
    """
    Creates unique directory to store simulation results.
    """
    uid = None
    if not directory:
        return uid, directory
    head, tail = os.path.split(directory)
    if tail == "auto":
        date = datetime.date.today().strftime("%Y-%m-%d") if not head else ""
        head = head or _RESULTCACHE
        for attempt in range(attempts):
            uid = uuid.uuid4().hex
            directory = os.path.join(os.path.expanduser(head), date, uid)
            if not os.path.isdir(directory):
                break
        assert (
            attempt + 1 < attempts
        ), f"Failed to generate unique id after {attempts} attempts"
    else:
        assert not os.path.isdir(directory), "Results directory already exists"
    os.makedirs(directory)
    return uid, directory


def _mkdir_resume(directory):
    """
    Returns the existing results directory when resuming.
    Does NOT create a new one.
    """
    assert os.path.isdir(directory), f"Results directory not found: {directory}"
    head, tail = os.path.split(directory)
    uid = tail  # reuse existing uid
    return uid, directory


def _storeresult(params, result):
    if params.simulation.results is None:
        return
    assert os.path.isdir(params.simulation.results)
    result.store(params.simulation.results)


def _storeparams(params, explorables=None):
    if params.simulation.results is None:
        return
    assert os.path.isdir(params.simulation.results)
    params.toJSON(params.simulation.results, filename="configuration.json")
    if explorables:
        fname = os.path.join(params.simulation.results, "exploration.json")
        with open(fname, "w") as fstream:
            json.dump(explorables, fstream, default=lambda x: x.__dict__, indent=4)


def _storegraph(params, graph, prefix):
    if not params.simulation.results:
        return
    assert os.path.isdir(params.simulation.results)
    fname = os.path.join(params.simulation.results, f"{prefix}.bin")
    dgl.save_graphs(fname, [graph])


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------

def _checkpoint_path(results_dir):
    """Returns the path to the checkpoint file."""
    return os.path.join(results_dir, _CHECKPOINT_FILE)


def _save_checkpoint(results_dir, repeat_idx, step, graph):
    """
    Saves current simulation state to a checkpoint file.

    Saves:
      - Which repeat we are on
      - Which step we are on within that repeat
      - Current beliefs tensor
      - Current reliability tensor (if present — randomly assigned per repeat)
    """
    ckpt = {
        "repeat_idx": repeat_idx,
        "step": step,
        "beliefs": graph.ndata["beliefs"].cpu(),
    }
    # Save reliability if present (randomly assigned in UnreliableOp)
    if "reliability" in graph.ndata:
        ckpt["reliability"] = graph.ndata["reliability"].cpu()

    path = _checkpoint_path(results_dir)
    # Write to a temp file first, then rename — avoids corruption if killed mid-write
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp_path, path)
    log.info(f"Checkpoint saved: repeat {repeat_idx + 1}, step {step}")


def _load_checkpoint(results_dir):
    """
    Loads checkpoint if it exists. Returns None if no checkpoint found.
    """
    path = _checkpoint_path(results_dir)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    log.info(
        f"Checkpoint found: resuming from repeat {ckpt['repeat_idx'] + 1}, "
        f"step {ckpt['step']}"
    )
    return ckpt


def _delete_checkpoint(results_dir):
    """Removes checkpoint file once simulation completes successfully."""
    path = _checkpoint_path(results_dir)
    if os.path.exists(path):
        os.remove(path)
        log.info("Checkpoint deleted — simulation complete.")


def _last_snapshot_step(hdf5_path):
    """
    Reads an existing snapshot hdf5 file and returns the last saved step number.
    Returns 0 if file doesn't exist or has no beliefs.
    """
    if not os.path.exists(hdf5_path):
        return 0
    with h5py.File(hdf5_path, "r") as f:
        if "beliefs" not in f:
            return 0
        steps = [int(k) for k in f["beliefs"].keys()]
        return max(steps) if steps else 0


# --------------------------------------------------------------------------

def random(seed=0):
    """
    Set random number generator for PolyGraph simulations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    dgl.random.seed(seed)


def explore(params, explorables):
    """
    Explores multiple PolyGraph configurations.
    """
    options = {var.name: var.values for var in explorables.values()}
    configurations = hparams.PolyGraphHyperParameters.expand(params, options)
    assert len(configurations) > 1
    assert params.simulation.results
    _, params.simulation.results = _mkdir(params.simulation.results)
    _storeparams(params, explorables=explorables)
    collection = collections.deque()
    for config in configurations:
        config.simulation.results = os.path.join(
            params.simulation.results, "explorations/auto"
        )
        meta = {key: config.getattr(var.name) for key, var in explorables.items()}
        log.info(
            "Explore {} ({} simulations)".format(
                ", ".join([f"{k} = {v}" for k, v in meta.items()]),
                config.simulation.repeats,
            )
        )
        result = simulate(config, **meta)
        collection.append(result)
    results = metadata.merge(*collection)
    _storeresult(params, results)
    return results


@torch.no_grad()
def simulate(params, op=None, **meta):
    """
    Runs a PolyGraph simulation multiple times, with checkpoint/resume support.

    If a checkpoint exists in the results directory, the simulation resumes
    from where it left off — same repeat, same step, same graph state.
    """
    assert isinstance(params, hparams.PolyGraphHyperParameters)
    if (params.op is None) == (op is None):
        if (params.op is None) and (op is None):
            raise ValueError("Operator not set")
        else:
            raise ValueError("Either params.op or op must be set, but not both")
    if op is None:
        op = ops.getbyname(params.op)
    else:
        params.op = op.__name__

    # -----------------------------------------------------------------------
    # Resume or fresh start
    # -----------------------------------------------------------------------
    # On a fresh run, params.simulation.results is either None or "auto" etc.
    # On a resume, the user passes the existing results directory via
    # POLYGRAPHS_RESUME env variable so we can find the checkpoint.
    resume_dir = os.getenv("POLYGRAPHS_RESUME")

    if resume_dir and os.path.isdir(resume_dir):
        # We are resuming — reuse the existing results directory
        uid, params.simulation.results = _mkdir_resume(resume_dir)
        log.info(f"Resuming simulation in: {params.simulation.results}")
    else:
        # Fresh start — create a new results directory
        uid, params.simulation.results = _mkdir(params.simulation.results)
        _storeparams(params)

    # Try to load a checkpoint
    ckpt = _load_checkpoint(params.simulation.results)

    # Which repeat to start from (0-indexed)
    start_repeat = ckpt["repeat_idx"] if ckpt else 0

    # Collection of simulation results
    results = metadata.PolyGraphSimulation(uid=uid, **meta)

    # -----------------------------------------------------------------------
    # Repeat loop
    # -----------------------------------------------------------------------
    for idx in range(params.simulation.repeats):

        # Skip repeats that were already completed in a previous run
        if idx < start_repeat:
            log.info(f"Skipping repeat #{idx + 1} (already completed)")
            continue

        log.debug("Simulation #{:04d} starts".format(idx + 1))

        # Prefix string for filenames, e.g. "01", "02", ...
        prefix = f"{(idx + 1):0{len(str(params.simulation.repeats))}d}"

        # Build the snapshot filename for this repeat
        hdf5_path = os.path.join(params.simulation.results, f"{prefix}.hd5")

        # Create a DGL graph with given configuration
        graph = graphs.create(params.network)
        graph = graph.to(device=params.device)

        # Create a model with given configuration (this randomly assigns reliability)
        model = op(graph, params)

        # ------------------------------------------------------------------
        # Restore graph state from checkpoint (if resuming this repeat)
        # ------------------------------------------------------------------
        if ckpt and idx == start_repeat:
            # Restore beliefs
            graph.ndata["beliefs"] = ckpt["beliefs"].to(device=params.device)
            # Restore reliability (was randomly assigned — must match original)
            if "reliability" in ckpt:
                graph.ndata["reliability"] = ckpt["reliability"].to(device=params.device)
                # Also restore on the model object so sampling is consistent
                if hasattr(model, "_reliability"):
                    model._reliability = ckpt["reliability"].to(device=params.device)
            # Find out which step to resume from
            resume_step = ckpt["step"]
            log.info(f"Restored graph state from checkpoint at step {resume_step}")
        else:
            resume_step = 0
            # Export initial graph (beliefs are initialised) — only on fresh repeat
            _storegraph(params, graph, prefix)

        model.eval()

        # Create hooks
        hooks = []
        if params.logging.enabled:
            hooks += [monitors.MonitorHook(interval=params.logging.interval)]
        if params.snapshots.enabled:
            hooks += [
                monitors.SnapshotHook(
                    interval=params.snapshots.interval,
                    messages=params.snapshots.messages,
                    location=params.simulation.results,
                    filename=f"{prefix}.hd5",
                )
            ]

        # Run simulation (with resume step and checkpoint saving)
        result = simulate_(
            graph,
            model,
            params,
            steps=params.simulation.steps,
            mistrust=params.mistrust,
            lowerupper=params.lowerupper,
            upperlower=params.upperlower,
            hooks=hooks,
            start_step=resume_step,
            repeat_idx=idx,
            results_dir=params.simulation.results,
            checkpoint_interval=100,  # save checkpoint every 100 steps
        )
        results.add(*result)
        log.info(
            "Sim #{:04d}: "
            "{:6d} steps "
            "{:7.2f}s; "
            "action: {:1s} "
            "undefined: {:<1} "
            "converged: {:<1} "
            "polarized: {:<1} ".format(idx + 1, *result)
        )

        # This repeat is done — update checkpoint to point to next repeat
        # so that if we die between repeats, we skip this one correctly
        _save_checkpoint(params.simulation.results, idx + 1, 0, graph)

    # All repeats done — remove checkpoint
    _delete_checkpoint(params.simulation.results)

    # Store simulation results
    _storeresult(params, results)
    return results


def simulate_(
    graph,
    model,
    params,
    steps=1,
    hooks=None,
    mistrust=0.0,
    lowerupper=0.5,
    upperlower=0.99,
    start_step=0,           # NEW: which step to resume from
    repeat_idx=0,           # NEW: which repeat we are in (for checkpoint)
    results_dir=None,       # NEW: where to save checkpoint
    checkpoint_interval=100 # NEW: save checkpoint every N steps
):
    """
    Runs a simulation either for a finite number of steps or until convergence.
    Supports resuming from a checkpoint via start_step.

    Returns:
        A 4-tuple that consists of (in order):
            a) number of simulation steps
            b) wall-clock time
            c) whether the network has converged or not
            d) whether the network is polarised or not
    """

    def cond(step):
        return step < steps if steps else True

    clock = timer.Timer()
    clock.start()

    # Resume from checkpoint step instead of 0
    step = start_step
    terminated = None

    while cond(step):
        step += 1

        # Update FactCheckersOp with current step
        if isinstance(model, ops.BaseFactCheckersOp):
            model.set_current_step(step)
            model.block(graph, params)

        # Forward operation on the graph
        _ = model(graph)

        # Monitor progress
        if hooks:
            for hook in hooks:
                hook.mayberun(step, graph)

        # Save checkpoint periodically
        if results_dir and step % checkpoint_interval == 0:
            _save_checkpoint(results_dir, repeat_idx, step, graph)

        # Check termination conditions
        terminated = (
            undefined(graph),
            converged(graph, upperlower=upperlower, lowerupper=lowerupper),
            polarized(
                graph, upperlower=upperlower, lowerupper=lowerupper, mistrust=mistrust
            ),
        )
        if any(terminated):
            break

    duration = clock.dt()
    if not terminated[0]:
        if hooks:
            for hook in hooks:
                hook.conclude(step, graph)
        act = consensus(graph, lowerupper=lowerupper)
    else:
        act = "?"

    return (
        step,
        duration,
        act,
    ) + terminated


def undefined(graph):
    belief = graph.ndata["beliefs"]
    result = torch.any(torch.isnan(belief)) or torch.any(torch.isinf(belief))
    return result.item()


def consensus(graph, lowerupper=0.99):
    if converged(graph, lowerupper=lowerupper):
        belief = graph.ndata["beliefs"]
        return "B" if torch.all(torch.gt(belief, lowerupper)) else "A"
    return "?"


def converged(graph, upperlower=0.5, lowerupper=0.99):
    tensor = graph.ndata["beliefs"]
    result = torch.all(torch.gt(tensor, lowerupper)) or torch.all(
        torch.le(tensor, upperlower)
    )
    return result.item()


def polarized(graph, mistrust=0.0, upperlower=0.5, lowerupper=0.99):
    if not mistrust:
        return False
    tensor = graph.ndata["beliefs"]
    c = torch.all(torch.gt(tensor, lowerupper) | torch.le(tensor, upperlower))
    b = torch.any(torch.gt(tensor, lowerupper))
    a = torch.any(torch.le(tensor, upperlower))
    if a and b and c:
        delta = torch.min(tensor[torch.gt(tensor, lowerupper)]) - torch.max(
            tensor[torch.le(tensor, upperlower)]
        )
        return torch.ge(delta * mistrust, 1).item()
    return False
  
