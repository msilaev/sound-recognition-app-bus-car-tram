from threading import Lock
class SingletonMeta(type):

    _instances = None
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instances or args or kwargs:
                instance = super().__call__(*args, **kwargs)
                cls._instances = instance
        return cls._instances

class Result(metaclass=SingletonMeta):

    _instance = None
    flag_first = True
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.flag_first = False
        else:
            cls.flag_first = True
        return cls._instance

    def __init__(self, result = "bus"):

            self.__result = result
    @property
    def result(self):
        return self.__result

    @result.setter
    def result(self, result):
        self.__result = result

    @property
    def height(self):
        return self.__height