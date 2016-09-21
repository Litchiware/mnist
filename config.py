class _const:
    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError, "Can't change const %s" %name
        if not name.isupper():
            raise self.ConstCaseError, 'const name "%s" is not all uppercase' %name
        self.__dict__[name] = value

import sys
sys.modules[__name__] = _const()

import config
import os

config.NUM_CHANNELS = 1
config.NUM_LABELS = 10
config.SEED = 66478
config.IMAGE_SIZE = 28
config.NUM_EPOCHS = 10
config.BATCH_SIZE = 64
config.WORK_DIRECTORY = '/home/llm/Datasets/MNIST'
config.SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
