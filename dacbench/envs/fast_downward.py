"""Planning environment from
"Learning Heuristic Selection with Dynamic Algorithm Configuration"
by David Speck, André Biedenkapp, Frank Hutter, Robert Mattmüller und Marius Lindauer.
Original environment authors: David Speck, André Biedenkapp.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from copy import deepcopy
from enum import Enum
from pathlib import Path

import numpy as np

from dacbench import AbstractEnv


class StateType(Enum):
    """Class to define numbers for state types."""

    RAW = 1
    DIFF = 2
    ABSDIFF = 3
    NORMAL = 4
    NORMDIFF = 5
    NORMABSDIFF = 6


class FastDownwardEnv(AbstractEnv):
    """Environment to control Solver Heuristics of FastDownward."""

    def __init__(self, config):
        """Initialize FD Env.

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super().__init__(config)
        self._heuristic_state_features = [
            "Average Value",  # 'Dead Ends Reliable',
            "Max Value",
            "Min Value",
            "Open List Entries",
            "Varianz",
        ]
        self._general_state_features = [
            # 'evaluated_states', 'evaluations', 'expanded_states',
            # 'generated_ops',
            # 'generated_states', 'num_variables',
            # 'registered_states', 'reopened_states',
            # "cg_num_eff_to_eff", "cg_num_eff_to_pre", "cg_num_pre_to_eff"
        ]

        total_state_features = len(config.heuristics) * len(
            self._heuristic_state_features
        )
        self._use_gsi = config.use_general_state_info
        if config.use_general_state_info:
            total_state_features += len(self._general_state_features)

        self.__skip_transform = [False for _ in range(total_state_features)]
        if config.use_general_state_info:
            self.__skip_transform[4] = True  # skip num_variables transform
            self.__skip_transform[7] = True
            self.__skip_transform[8] = True
            self.__skip_transform[9] = True

        self.heuristics = config.heuristics
        self.host = config.host
        self._port = config.get("port", 0)
        if config["parallel"]:
            self.port = 0

        self.fd_seed = config.fd_seed
        self.control_interval = config.control_interval

        if config.fd_logs is None:
            self.logpath_out = os.devnull
            self.logpath_err = os.devnull
        else:
            self.logpath_out = Path(config.fd_logs) / "fdout.txt"
            self.logpath_err = Path(config.fd_logs) / "fderr.txt"
        self.fd_path = config.fd_path
        self.fd = None
        if "domain_file" in config:
            self.domain_file = config["domain_file"]

        self.socket = None
        self.conn = None

        self._prev_state = None
        self.num_steps = config.num_steps

        self.__state_type = StateType(config.state_type)
        self.__norm_vals = []
        self._config_dir = config.config_dir
        self._port_file_id = config.port_file_id

        self._transformation_func = None
        # create state transformation function with inputs
        # (current state, previous state, normalization values)
        if self.__state_type == StateType.DIFF:
            self._transformation_func = lambda x, y, z, skip: x - y if not skip else x
        elif self.__state_type == StateType.ABSDIFF:
            self._transformation_func = lambda x, y, z, skip: (
                abs(x - y) if not skip else x
            )
        elif self.__state_type == StateType.NORMAL:
            self._transformation_func = lambda x, y, z, skip: (
                FastDownwardEnv._save_div(x, z) if not skip else x
            )
        elif self.__state_type == StateType.NORMDIFF:
            self._transformation_func = lambda x, y, z, skip: (
                FastDownwardEnv._save_div(x, z) - FastDownwardEnv._save_div(y, z)
                if not skip
                else x
            )
        elif self.__state_type == StateType.NORMABSDIFF:
            self._transformation_func = lambda x, y, z, skip: (
                abs(FastDownwardEnv._save_div(x, z) - FastDownwardEnv._save_div(y, z))
                if not skip
                else x
            )

        self.max_rand_steps = config.max_rand_steps
        self.__start_time = None
        self.done = True  # Starts as true as the expected behavior is that
        # before normal resets an episode was done.

    @property
    def port(self):
        """Port function."""
        if self._port == 0:
            if self.socket is None:
                raise ValueError(
                    "Automatic port selection enabled. Port not know at the moment"
                )
            _, port = self.socket.getsockname()
        else:
            port = self._port
        return port

    @port.setter
    def port(self, port):
        self._port = port

    @property
    def _argstring(self):
        # if a socket is bound to 0 it will automatically choose a free port
        return (
            f"rl_eager(rl([{''.join(f'{h},' for h in self.heuristics)[:-1]}],"
            f"random_seed={self.fd_seed}),rl_control_interval={self.control_interval},rl_client_port={self.port})"
        )

    @staticmethod
    def _save_div(a, b):
        """Helper method for safe division.

        Parameters
        ----------
        a : list or np.array
            values to be divided
        b : list or np.array
            values to divide by

        Returns:
        -------
        np.array
            Division result
        """
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def send_msg(self, msg: bytes):
        """Send message and prepend the message size.

        Based on comment from SO see [1]
        [1] https://stackoverflow.com/a/17668009

        Parameters
        ----------
        msg : bytes
            The message as byte
        """
        # Prefix each message with a 4-byte length (network byte order)
        msg = str.encode(f"{len(msg):>04d}") + msg
        self.conn.sendall(msg)

    def recv_msg(self):
        """Recieve a whole message. The message has to be prepended with its total size
        Based on comment from SO see [1].

        Returns:
        ----------
        bytes
            The message as byte
        """
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = int(raw_msglen.decode())
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n: int):
        """Given we know the size we want to recieve,
        we can recieve that amount of bytes.
        Based on comment from SO see [1].

        Parameters
        ---------
        n: int
            Number of bytes to expect in the data

        Returns:
        ----------
        bytes
            The message as byte
        """
        # Helper function to recv n bytes or return None if EOF is hit
        data = b""
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _process_data(self):
        """Split received json into state reward and done.

        Returns:
        ----------
        np.array, float, bool
            state, reward, done
        """
        msg = self.recv_msg().decode()
        # print("----------------------------")
        # print(msg)
        # print("=>")
        msg = msg.replace("-inf", "0")
        msg = msg.replace("inf", "0")
        # print(msg)
        data = eval(msg)  # noqa: S307
        r = data["reward"]
        done = data["done"]
        del data["reward"]
        del data["done"]

        state = []

        if self._use_gsi:
            for feature in self._general_state_features:
                state.append(data[feature])
        for heuristic_id in range(len(self.heuristics)):  # process heuristic data
            for feature in self._heuristic_state_features:
                state.append(data["%d" % heuristic_id][feature])

        if self._prev_state is None:
            self.__norm_vals = deepcopy(state)
            self._prev_state = deepcopy(state)
        if (
            self.__state_type != StateType.RAW
        ):  # Transform state to DIFF state or normalize
            tmp_state = state
            state = list(
                map(
                    self._transformation_func,
                    state,
                    self._prev_state,
                    self.__norm_vals,
                    self.__skip_transform,
                )
            )
            self._prev_state = tmp_state
        return np.array(state), r, done

    def step(self, action: int | list[int]):
        """Environment step.

        Parameters
        ---------
        action: typing.Union[int, List[int]]
            Parameter(s) to apply

        Returns:
        ----------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, info
        """
        self.done = super().step_()
        if not np.issubdtype(
            type(action), np.integer
        ):  # check for core int and any numpy-int
            try:
                action = action[0]
            except IndexError as e:
                print(type(action))
                raise e
        if self.num_steps:
            msg = ",".join([str(action), str(self.num_steps)])
        else:
            msg = str(action)
        self.send_msg(str.encode(msg))
        s, r, terminated = self._process_data()
        r = max(self.reward_range[0], min(self.reward_range[1], r))
        info = {}
        if terminated:
            self.done = True
            self.kill_connection()
        if self.c_step > self.n_steps:
            info["needs_reset"] = True
            self.send_msg(str.encode("END"))
            self.kill_connection()
        return s, r, terminated, self.done, info

    def reset(self, seed=None, options=None):
        """Reset environment.

        Returns:
        ----------
        np.array
            State after reset
        dict
            Meta-info
        """
        if options is None:
            options = {}
        super().reset_(seed)
        self._prev_state = None
        self.__start_time = time.time()
        if not self.done:  # This means we interrupt FD before a plan was found
            # Inform FD about imminent shutdown of the connection
            self.send_msg(str.encode("END"))
        self.done = False
        if self.conn:
            self.conn.shutdown(2)
            self.conn.close()
            self.conn = None
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(10)
            self.socket.bind((self.host, self.port))

        if self.fd:
            self.fd.terminate()

        if self.instance.endswith(".pddl"):
            command = [
                "python3",
                f"{self.fd_path}",
                self.domain_file,
                self.instance,
                "--search",
                self._argstring,
            ]
        else:
            command = [
                "python3",
                f"{self.fd_path}",
                self.instance,
                "--search",
                self._argstring,
            ]

        with open(self.logpath_out, "a+") as fout, open(self.logpath_err, "a+") as ferr:
            err_output = subprocess.STDOUT if self.logpath_err == "/dev/null" else ferr
            self.fd = subprocess.Popen(command, stdout=fout, stderr=err_output)  # noqa: S603

        # write down port such that FD can potentially read where to connect to
        if self._port_file_id:
            fp = Path(self._config_dir) / f"port_{self._port_file_id:d}.txt"
        else:
            fp = Path(self._config_dir) / f"port_{self.port}.txt"
        with open(fp, "w") as portfh:
            portfh.write(str(self.port))

        self.socket.listen()
        try:
            self.conn, address = self.socket.accept()
        except TimeoutError:
            raise OSError(  # noqa: B904
                "Fast downward subprocess not reachable (time out). "
                "Possible solutions:\n"
                " (1) Did you run './dacbench/envs/rl-plan/fast-downward/build.py' "
                "in order to build the fd backend?\n"
                " (2) Try to fix this by setting OPENBLAS_NUM_THREADS=1. "
                "For more details see https://github.com/automl/DACBench/issues/96"
            )

        s, _, _ = self._process_data()
        if self.max_rand_steps > 1:
            for _ in range(self.np_random.randint(1, self.max_rand_steps + 1)):
                s, _, _, _, _ = self.step(self.action_space.sample())
                if self.conn is None:
                    return self.reset()
        else:
            s, _, _, _, _ = self.step(0)  # hard coded to zero as initial step

        Path.unlink(
            fp
        )  # remove the port file such that there is no chance of loading the old port
        return s, {}

    def kill_connection(self):
        """Kill the connection."""
        if self.conn:
            self.conn.shutdown(2)
            self.conn.close()
            self.conn = None
        if self.socket:
            self.socket.shutdown(2)
            self.socket.close()
            self.socket = None

    def close(self):
        """Close Env.

        Returns:
        -------
        bool
            Closing confirmation
        """
        if self.socket is None:
            return True
        fp = Path(self._config_dir) / f"port_{self.port}.txt"
        if Path.exists(fp):
            Path.unlink(fp)

        self.kill_connection()
        return True

    def render(self, mode: str = "human") -> None:
        """Required by gym.Env but not implemented.

        Parameters
        -------
        mode : str
            Rendering mode
        """
