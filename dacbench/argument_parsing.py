"""Argument parsing."""

from __future__ import annotations

from argparse import ArgumentTypeError as Err
from pathlib import Path


class PathType:
    """Custom argument type for path validation.

    Adapted from: https://stackoverflow.com/questions/11415570/directory-path-types-with-argparse
    """

    def __init__(self, exists=True, file_type="file", dash_ok=True):
        """Initialize Path.

        Parameters
        ----------
        exists : bool
            True: a path that does exist
            False: a path that does not exist, in a valid parent directory
            None: don't care
        file_type :  str
            file, dir, symlink, socket, None, or a function returning True
            for valid paths
            None: don't care
        dash_ok: bool
            whether to allow "-" as stdin/stdout

        """
        assert exists in (True, False, None)
        assert file_type in ("file", "dir", "symlink", "socket", None) or callable(
            file_type
        )

        self._exists = exists
        self._file_type = file_type
        self._dash_ok = dash_ok

    def __call__(self, string: str):
        """Call Path.

        Parameters
        ----------
        string : str
            string to check

        """
        if string == "-":
            # the special argument "-" means sys.std{in,out}
            if self._file_type == "dir":
                raise Err("standard input/output (-) not allowed as directory path")
            if self._file_type == "symlink":
                raise Err("standard input/output (-) not allowed as symlink path")
            if not self._dash_ok:
                raise Err("standard input/output (-) not allowed")

        path = Path(string)

        # existence
        if self._exists is None:
            pass
        elif self._exists != path.exists():
            negate = "" if self._exists else "not"
            positive = "" if not self._exists else "not"
            raise Err(
                f"{self._file_type.capitalize()} should {negate} "
                f"exist but does {positive}"
            )

            # type
            if self._file_type is None:
                pass
            elif isinstance(self._file_type, str):
                check = getattr(path, f"is_{self._file_type}")
                if not check():
                    raise Err(f"Path is not {self._file_type}")
            elif isinstance(self._file_type, callable):
                if not self._file_type(path):
                    raise Err("Callable type check failed")
            else:
                raise Err("invalid type to check for")

        return path
