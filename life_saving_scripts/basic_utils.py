import numpy as np
from typing import Callable, Union, Any, Optional, List
import logging
import os
import json
from inspect import signature


A = np.ndarray
END = "\033[0m"
BOLD = "\033[1m"
BROWN = "\033[0;33m"
ITALIC = "\033[3m"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def set_logger_level_to_all_local(level: int) -> None:
    """Sets the level of all local loggers to the given level.

    Parameters
    ----------
    level : int, optional
        The level to set the loggers to, by default logging.DEBUG.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]

    for _, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            if hasattr(logger, "local"):
                logger.setLevel(level)


def create_simple_logger(
    logger_name: str, level: str = LOG_LEVEL, set_level_to_all_loggers: bool = False
) -> logging.Logger:
    """Creates a simple logger with the given name and level. The logger has a single handler that logs to the console.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    level : str or int
        Level of the logger. Can be a string or an integer. If a string, it should be one of the following: "debug", "info", "warning", "error", "critical". Default level is read from the environment variable LOG_LEVEL.

    Returns
    -------
    logging.Logger
        The logger object.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(logger_name)
    logger.local = True
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if set_level_to_all_loggers:
        set_logger_level_to_all_local(level)
    return logger


logger = create_simple_logger(__name__)


def is_jupyter_notebook() -> bool:
    """Checks if the code is being run in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is being run in a Jupyter notebook, False otherwise.
    """
    is_jupyter = False
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython

        # noinspection PyUnresolvedReferences
        if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
            pass
        else:
            is_jupyter = True
    except (ImportError, NameError):
        pass
    if is_jupyter:
        logger.debug("Running in Jupyter notebook.")
    else:
        logger.debug("Not running in a Jupyter notebook.")
    return is_jupyter


def get_parameters_list(func: Callable) -> tuple[list[str], list[str]]:
    """Gets the list of non-optional and optional parameters of a function.

    Parameters
    ----------
    func : Callable
        The function to get the parameters of.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing the list of non-optional and optional parameters.
    """
    optional_params = [
        p.name for p in signature(func).parameters.values() if p.default != p.empty
    ]
    non_optional_params = [
        p.name for p in signature(func).parameters.values() if p.default == p.empty
    ]
    # remove special parameters like self
    if "self" in non_optional_params:
        non_optional_params.remove("self")
    return non_optional_params, optional_params


def create_wandb_logger(
    name: Union[str, None] = None,
    project: Union[str, None] = None,
    config: Union[dict[str, any], None] = None,
    tags: Union[list[str], None] = None,
    notes: str = "",
    group: Union[str, None] = None,
    job_type: str = "",
    logger: Union[logging.Logger, None] = None,
) -> Any:
    """Creates a new run on Weights & Biases and returns the run object.

    Parameters
    ----------
    project : str | None, optional
        The name of the project. If None, it must be provided in the config. Default is None.
    name : str | None, optional
        The name of the run. If None, it must be provided in the config. Default is None.
    config : dict[str, any] | None, optional
        The configuration to be logged. Default is None. If `project` and `name` are not provided, they must be present in the config.
    tags : list[str] | None, optional
        The tags to be added to the run. Default is None.
    notes : str, optional
        The notes to be added to the run. Default is "".
    group : str | None, optional
        The name of the group to which the run belongs. Default is None.
    job_type : str, optional
        The type of job. Default is "train".
    logger : logging.Logger | None, optional
        The logger to be used by the object. If None, a simple logger is created using `create_simple_logger`. Default is None.

    Returns
    -------
    wandb.Run
        The run object.
    """
    import wandb

    logger = logger or create_simple_logger("create_wandb_logger")
    if config is None:
        logger.debug("No config provided. Using an empty config.")
        config = {}

    if name is None and "name" not in config.keys():
        m = "Run name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    if project is None and "project" not in config.keys():
        m = "Project name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    # If the arguments are provided, they take precedence over the config
    name = name or config.get("name")
    project = project or config.get("project")
    notes = notes or config.get("notes")
    tags = tags or config.get("tags")
    group = group or config.get("group")
    job_type = job_type or config.get("job_type")

    logger.info(
        f"Initializing Weights & Biases for project {project} with run name {name}."
    )
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
    )
    return wandb


class Config:
    """An abstract class for configuration classes. A configuration class has two main attributes:

    - ALLOWED_KEYS: a list of allowed keys
    - ConfigFor: the class that this config is for

    Using this class, we can create a configuration class for any class by inheriting from it and setting the ALLOWED_KEYS and ConfigFor attributes.
    """

    ALLOWED_KEYS: List[str] = None  # a list of allowed keys
    ConfigFor: Union[object, str, None] = None  # the class that this config is for

    def __init__(self, **kwargs):

        # check if all the keys are allowed
        keys_provided = list(kwargs.keys())
        if self.ALLOWED_KEYS is None:
            delta = []
        else:
            delta = list(set(keys_provided) - set(self.ALLOWED_KEYS))
        if len(delta) > 0:
            msg = f"Provided keys not allowed: {delta}"
            logger.error(msg)
            raise ValueError(msg)

        self.__dict__.update(kwargs)

    def to_dict(self):
        if self.ALLOWED_KEYS is None:
            return self.__dict__

        all_variables = list(self.__dict__.keys())
        available_keys = self.ALLOWED_KEYS
        intersection = list(set(all_variables) & set(available_keys))
        return {k: self.__dict__[k] for k in intersection}

    @staticmethod
    def load_from_path(path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return Config(**d)

    def load_object(self, obj: Optional[object] = None):
        # if ConfigFor is not set, and obj is not provided, we can't create the object
        if self.ConfigFor is None and obj is None:
            msg = "ConfigFor is not set, and obj is not provided"
            logger.error(msg)
            raise ValueError(msg)

        # if no object is provided, we will use ConfigFor
        if obj is None:
            # if ConfigFor is a string, we will use the global object with that name
            if isinstance(self.ConfigFor, str):
                logger.info("No object provided, Getting object from globals")
                obj = globals()[self.ConfigFor]
            else:
                logger.info("No object provided, using ConfigFor")
                obj = self.ConfigFor
            logger.debug(f"Found object: {obj.__class__.__name__}")

        # check if all non-optional parameters are available before creating the object
        non_optional_params, optional_params = get_parameters_list(obj.__init__)
        params_dict = self.to_dict()
        parameters_available = list(params_dict.keys())

        delta_non_optional = list(set(non_optional_params) - set(parameters_available))
        if len(delta_non_optional) > 0:
            msg = f"Missing non-optional parameters: {delta_non_optional}"
            logger.error(msg)
            raise ValueError(msg)

        # for optional parameters, we can ignore them if they are not available
        delta_optional = list(set(optional_params) - set(parameters_available))
        if len(delta_optional) > 0:
            msg = f"Missing optional parameters: {delta_optional}"
            logger.warning(msg)

        return obj(**self.to_dict())

    def save(self, path: str):
        # not that this may always work for some classes
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def to_dict_serializable(self):
        """Converts the configuration to a dictionary that can be serialized to JSON."""
        params = self.to_dict().copy()
        for k, v in params.items():
            if isinstance(v, Config):
                params[k] = v.to_dict_serializable()
            if not isinstance(v, (int, float, str, list, dict)):
                try:
                    logger.warning(
                        f"Non-serializable value found for {k}. Only class name will be saved."
                    )
                    params[k] = v.__class__.__name__
                except Exception as e:
                    logger.warning(
                        f"Error getting class name for {k}. It will be kept empty."
                    )
                    params[k] = None
            else:
                params[k] = v

        return params


def set_publish_plotly_template() -> None:
    """Sets the plotly template for publication-ready plots."""
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "notebook"
    font_family = "Times New Roman"

    def get_font_dict(size, color="black"):
        return dict(
            size=size,
            color=color,
            family=font_family,
            weight="bold",
            variant="small-caps",
        )

    pio.templates["publish"] = go.layout.Template(
        layout=go.Layout(
            title=dict(
                font=get_font_dict(24),
            ),
            legend=dict(
                font=get_font_dict(18),
            ),
            xaxis=dict(
                title=dict(
                    font=get_font_dict(18),
                ),
                tickfont=get_font_dict(16),
            ),
            yaxis=dict(
                title=dict(
                    font=get_font_dict(18),
                ),
                tickfont=get_font_dict(16),
            ),
        )
    )
    pio.templates.default = "publish"
    logger.info("Plotly template ready for publication.")


def set_publish_matplotlib_template() -> None:
    """Sets the matplotlib template for publication-ready plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelcolor": "#000000",
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.grid": True,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "figure.titlesize": 20,
            "figure.titleweight": "bold",
            "grid.color": "#D3D3D3",
            "grid.linewidth": 1.5,
            "grid.linestyle": "--",
        }
    )
    logger.info("Matplotlib template ready for publication.")
