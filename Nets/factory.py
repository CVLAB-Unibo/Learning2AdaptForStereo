_FACTORY = {}

def getStereoNet(name,args):
    if name not in _FACTORY:
        raise Exception('Unrecognized network name: {}'.format(name))
    else:
        return _FACTORY[name](**args)

def checkExistance(name):
    return name in _FACTORY.keys()

def getAvailableNets():
    return _FACTORY.keys()

def register_net_to_factory():
    def decorator(cls):
        _FACTORY[cls._name] = cls
        return cls
    return decorator