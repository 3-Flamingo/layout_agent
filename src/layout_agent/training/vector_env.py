"""Small sync/subprocess vector wrappers for LayoutEnv.

The subprocess implementation uses ``spawn`` so behavior matches Windows and
CUDA hosts. Workers auto-reset completed episodes and preserve final info.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from layout_agent.training.env import LayoutEnv


def _stack(observations: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def _worker(
    connection: Any,
    state_paths: List[str],
    env_kwargs: Dict[str, Any],
    worker_id: int,
    seed: int,
) -> None:
    try:
        kwargs = dict(env_kwargs)
        work = Path(kwargs.pop("work_dir"))
        env = LayoutEnv(
            [Path(path) for path in state_paths],
            work_dir=work / f"worker_{worker_id:03d}",
            seed=seed + worker_id,
            **kwargs,
        )
        episode_seed = seed + worker_id
        while True:
            command, payload = connection.recv()
            if command == "reset":
                reset_seed = int(payload) if payload is not None else episode_seed
                observation, info = env.reset(seed=reset_seed)
                connection.send(("ok", (observation, info)))
            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(int(payload))
                if terminated or truncated:
                    final_info = info
                    episode_seed += 10007
                    observation, reset_info = env.reset(seed=episode_seed)
                    info = {
                        "final_info": final_info,
                        "reset_info": reset_info,
                        "action": final_info.get("action"),
                        "no_op": final_info.get("no_op", False),
                        "rolled_back": final_info.get("rolled_back", False),
                        "step_time_ms": final_info.get("step_time_ms", 0.0),
                    }
                connection.send(
                    (
                        "ok",
                        (observation, reward, terminated, truncated, info),
                    )
                )
            elif command == "close":
                env.close()
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown vector command {command!r}")
    except BaseException as exc:  # noqa: BLE001
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except BaseException:  # noqa: BLE001
            pass
    finally:
        connection.close()


class SubprocessLayoutVectorEnv:
    def __init__(
        self,
        state_paths: Sequence[Path | str],
        num_envs: int,
        env_kwargs: Dict[str, Any],
        *,
        seed: int = 17,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        context = mp.get_context("spawn")
        self.connections: List[Any] = []
        self.processes: List[mp.Process] = []
        paths = [str(Path(path)) for path in state_paths]
        for worker_id in range(num_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, paths, env_kwargs, worker_id, seed),
                daemon=True,
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        self.num_envs = num_envs
        self.closed = False

    @staticmethod
    def _receive(connection: Any) -> Any:
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(
                f"vector worker {payload['type']}: {payload['message']}\n"
                f"{payload['traceback']}"
            )
        return payload

    def reset(
        self,
        seeds: Sequence[int] | None = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError("seeds length must equal num_envs")
        for index, connection in enumerate(self.connections):
            connection.send(("reset", None if seeds is None else int(seeds[index])))
        results = [self._receive(connection) for connection in self.connections]
        observations, infos = zip(*results)
        return _stack(observations), list(infos)

    def step(
        self,
        actions: Sequence[int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Dict[str, Any]],
    ]:
        if len(actions) != self.num_envs:
            raise ValueError("actions length must equal num_envs")
        for connection, action in zip(self.connections, actions):
            connection.send(("step", int(action)))
        results = [self._receive(connection) for connection in self.connections]
        observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            _stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            list(infos),
        )

    def close(self) -> None:
        if self.closed:
            return
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self.connections:
            try:
                self._receive(connection)
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self.closed = True

    def __enter__(self) -> "SubprocessLayoutVectorEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SyncLayoutVectorEnv:
    def __init__(
        self,
        state_paths: Sequence[Path | str],
        num_envs: int,
        env_kwargs: Dict[str, Any],
        *,
        seed: int = 17,
    ):
        kwargs = dict(env_kwargs)
        work = Path(kwargs.pop("work_dir"))
        self.envs = [
            LayoutEnv(
                state_paths,
                work_dir=work / f"worker_{index:03d}",
                seed=seed + index,
                **kwargs,
            )
            for index in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(
        self,
        seeds: Sequence[int] | None = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        results = [
            env.reset(seed=(None if seeds is None else int(seeds[index])))
            for index, env in enumerate(self.envs)
        ]
        observations, infos = zip(*results)
        return _stack(observations), list(infos)

    def step(
        self,
        actions: Sequence[int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Dict[str, Any]],
    ]:
        observations: List[Dict[str, np.ndarray]] = []
        rewards: List[float] = []
        terms: List[bool] = []
        truncs: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for env, action in zip(self.envs, actions):
            observation, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                final_info = info
                observation, reset_info = env.reset()
                info = {
                    "final_info": final_info,
                    "reset_info": reset_info,
                    "action": final_info.get("action"),
                    "no_op": final_info.get("no_op", False),
                    "rolled_back": final_info.get("rolled_back", False),
                    "step_time_ms": final_info.get("step_time_ms", 0.0),
                }
            observations.append(observation)
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos.append(info)
        return (
            _stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=np.bool_),
            np.asarray(truncs, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __enter__(self) -> "SyncLayoutVectorEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
