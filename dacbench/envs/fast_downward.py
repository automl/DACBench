"""
Planning environment from
"Learning Heuristic Selection with Dynamic Algorithm Configuration"
by David Speck, André Biedenkapp, Frank Hutter, Robert Mattmüller und Marius Lindauer.
Original environment authors: David Speck, André Biedenkapp
"""

import socket
import time
import typing
from copy import deepcopy
from enum import Enum
from os import remove
from os.path import join as joinpath
import subprocess

import numpy as np
from dacbench import AbstractEnv


class StateType(Enum):
    """Class to define numbers for state types"""

    RAW = 1
    DIFF = 2
    ABSDIFF = 3
    NORMAL = 4
    NORMDIFF = 5
    NORMABSDIFF = 6


class FastDownwardEnv(AbstractEnv):
    def __init__(self, config):
        """
        Initialize environment
        """

        super(FastDownwardEnv, self).__init__(config)
        self._heuristic_state_features = [
            "Average Value",  # 'Dead Ends Reliable',
            "Max Value",
            "Min Value",
            "Open List Entries",
            "Varianz",
        ]
        self._general_state_features = [  # 'evaluated_states', 'evaluations', 'expanded_states',
            # 'generated_ops',
            # 'generated_states', 'num_variables',
            # 'registered_states', 'reopened_states',
            # "cg_num_eff_to_eff", "cg_num_eff_to_pre", "cg_num_pre_to_eff"
        ]

        total_state_features = config.num_heuristics * len(
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

        self.__num_heuristics = config.num_heuristics
        self.host = config.host
        self.port = config.port
        if config["parallel"]:
            self.port += np.random.randint(50)
        self.fd_path = config.fd_path
        self.fd = None
        if "domain_file" in config.keys():
            self.domain_file = config["domain_file"]

        self.socket = None
        self.conn = None

        self._prev_state = None
        self.num_steps = config.num_steps
        self.control_interval = config.control_interval
        self.fd_seed = config.fd_seed

        self.__state_type = StateType(config.state_type)
        self.__norm_vals = []
        self._config_dir = config.config_dir
        self._port_file_id = config.port_file_id

        self._transformation_func = None
        # create state transformation function with inputs (current state, previous state, normalization values)
        if self.__state_type == StateType.DIFF:
            self._transformation_func = lambda x, y, z, skip: x - y if not skip else x
        elif self.__state_type == StateType.ABSDIFF:
            self._transformation_func = (
                lambda x, y, z, skip: abs(x - y) if not skip else x
            )
        elif self.__state_type == StateType.NORMAL:
            self._transformation_func = (
                lambda x, y, z, skip: FastDownwardEnv._save_div(x, z) if not skip else x
            )
        elif self.__state_type == StateType.NORMDIFF:
            self._transformation_func = (
                lambda x, y, z, skip: FastDownwardEnv._save_div(x, z)
                - FastDownwardEnv._save_div(y, z)
                if not skip
                else x
            )
        elif self.__state_type == StateType.NORMABSDIFF:
            self._transformation_func = (
                lambda x, y, z, skip: abs(
                    FastDownwardEnv._save_div(x, z) - FastDownwardEnv._save_div(y, z)
                )
                if not skip
                else x
            )

        self.rng = np.random.RandomState(seed=config.seed)
        self.max_rand_steps = config.max_rand_steps
        self.__start_time = None
        self.done = True  # Starts as true as the expected behavior is that before normal resets an episode was done.

    @staticmethod
    def _save_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def send_msg(self, msg: bytes):
        """
        Send message and prepend the message size

        Based on comment from SO see [1]
        [1] https://stackoverflow.com/a/17668009

        Parameters
        ----------
        msg : bytes
            The message as byte
        """
        # Prefix each message with a 4-byte length (network byte order)
        msg = str.encode("{:>04d}".format(len(msg))) + msg
        self.conn.sendall(msg)

    def recv_msg(self):
        """
        Recieve a whole message. The message has to be prepended with its total size
        Based on comment from SO see [1]

        Returns
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
        """
        Given we know the size we want to recieve, we can recieve that amount of bytes.
        Based on comment from SO see [1]

        Parameters
        ---------
        n: int
            Number of bytes to expect in the data

        Returns
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
        """
        Split received json into state reward and done

        Returns
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
        data = eval(msg)
        r = data["reward"]
        done = data["done"]
        del data["reward"]
        del data["done"]

        state = []

        if self._use_gsi:
            for feature in self._general_state_features:
                state.append(data[feature])
        for heuristic_id in range(self.__num_heuristics):  # process heuristic data
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

    def step(self, action: typing.Union[int, typing.List[int]]):
        """
        Environment step

        Parameters
        ---------
        action: typing.Union[int, List[int]]
            Parameter(s) to apply

        Returns
        ----------
        np.array, float, bool, dict
            state, reward, done, info
        """
        self.done = super(FastDownwardEnv, self).step_()
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
        s, r, d = self._process_data()
        r= max(self.reward_range[0], min(self.reward_range[1], r))
        info = {}
        if d:
            self.done = True
            self.kill_connection()
        if self.c_step > self.n_steps:
            info["needs_reset"] = True
            self.send_msg(str.encode("END"))
            self.kill_connection()
        return s, r, d or self.done, info

    def reset(self):
        """
        Reset environment

        Returns
        ----------
        np.array
            State after reset
        """
        super(FastDownwardEnv, self).reset_()
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
            self.socket.settimeout(0.5)
            self.socket.bind((self.host, self.port))

        if self.fd:
            self.fd.terminate()
        if self.instance.endswith(".pddl"):
            self.fd = subprocess.Popen(
                [
                    "python3",
                    f"{self.fd_path}",
                    self.domain_file,
                    self.instance,
                    "--search",
                    f"rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],random_seed={self.fd_seed}),rl_control_interval={self.control_interval},rl_client_port={self.port})",
                ]
            )
        else:
            self.fd = subprocess.Popen(
                [
                    "python3",
                    f"{self.fd_path}",
                    self.instance,
                    "--search",
                    f"rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],random_seed={self.fd_seed}),rl_control_interval={self.control_interval},rl_client_port={self.port})",
                ]
            )
        # write down port such that FD can potentially read where to connect to
        if self._port_file_id:
            fp = joinpath(self._config_dir, "port_{:d}.txt".format(self._port_file_id))
        else:
            fp = joinpath(self._config_dir, f"port_{self.port}.txt")
        with open(fp, "w") as portfh:
            portfh.write(str(self.port))
        print(fp)

        self.socket.listen()
        self.conn, address = self.socket.accept()
        s, _, _ = self._process_data()
        if self.max_rand_steps > 1:
            for _ in range(self.rng.randint(1, self.max_rand_steps + 1)):
                s, _, _, _ = self.step(self.action_space.sample())
        else:
            s, _, _, _ = self.step(0)  # hard coded to zero as initial step

        remove(
            fp
        )  # remove the port file such that there is no chance of loading the old port
        return s

    def kill_connection(self):
        """Kill the connection"""
        if self.conn:
            self.conn.shutdown(2)
            self.conn.close()
            self.conn = None
        if self.socket:
            self.socket.shutdown(2)
            self.socket.close()
            self.socket = None

    def close(self):
        """Needs to "kill" the environment"""
        self.kill_connection()
        return True

    def render(self, mode: str = "human") -> None:
        """
        Required by gym.Env but not implemented
        :param mode:
        :return: None
        """
        pass