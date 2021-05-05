from abc import ABC, abstractmethod
import logging
from uuid import uuid4


class Transformer(ABC):
    """ A Transformer class is a class with a public method transform
    that takes a dictionary of data and returns some transformation
    of that data.
    """
    def __init__(self):
        self._set_up_logging()

    def __call__(self, *args, **kwargs):
        self.log.info('Applying {}'.format(self.__class__.__name__))
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, data):
        pass

    def _set_up_logging(self):
        class_name = self.__class__.__name__
        self.log = logging.getLogger(class_name + str(uuid4()))
        self.log.setLevel('INFO')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s:::{}:::%(asctime)s:::%(message)s'.format(class_name))
        handler.setFormatter(formatter)
        self.log.addHandler(handler)


class Preprocessor(Transformer):
    """ A Preprocessor class is a Transformer class whose transform
    method takes a dictionary of data and returns that same
    dictionary after some modification. The Preprocessor overwrites.
    """
    pass


class Feature(Transformer):
    """ A Feature class is a Transformer class whose transform
    method takes a dictionary of data and returns a new
    data object. The Feature creates something new respecting
    the existing data.
    """
    pass
