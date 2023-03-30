from argparse import ArgumentTypeError as err
from pathlib import Path


class PathType(object):
    """
    Custom argument type for path validation.

    Adapted from: https://stackoverflow.com/questions/11415570/directory-path-types-with-argparse
    """

    def __init__(self, exists=True, type="file", dash_ok=True):
        """
        Initialize Path.

        Parameters
        ----------
        exists : bool
            True: a path that does exist
            False: a path that does not exist, in a valid parent directory
            None: don't care
        type :  str
            file, dir, symlink, socket, None, or a function returning True for valid paths
            None: don't care
        dash_ok: bool
            whether to allow "-" as stdin/stdout

        """
        assert exists in (True, False, None)
        assert type in ("file", "dir", "symlink", "socket", None) or hasattr(
            type, "__call__"
        )

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string: str):
        """
        Call Path.

        Parameters
        ----------
        string : str
            string to check

        """
        if string == "-":
            # the special argument "-" means sys.std{in,out}
            if self._type == "dir":
                raise err("standard input/output (-) not allowed as directory path")
            elif self._type == "symlink":
                raise err("standard input/output (-) not allowed as symlink path")
            elif not self._dash_ok:
                raise err("standard input/output (-) not allowed")

        path = Path(string)

        # existence
        if self._exists is None:
            pass
        elif not self._exists == path.exists():
            negate = "" if self._exists else "not"
            positive = "" if not self._exists else "not"
            raise err(
                f"{self._type.capitalize()} should {negate} exist but does {positive}"
            )

            # type
            if self._type is None:
                pass
            elif isinstance(self._type, str):
                check = getattr(path, f"is_{self._type}")
                if not check():
                    raise err(f"Path is not {self._type}")
            elif isinstance(self._type, callable):
                if not self._type(path):
                    raise err("Callable type check failed")
            else:
                raise err("invalid type to check for")

        return path
