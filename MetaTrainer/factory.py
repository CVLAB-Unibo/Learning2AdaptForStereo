_META_TRAINER_FACTORY = {}

def get_meta_learner(name,args):
    """
    Return a meta trainer given the name and the proper args for creation
    """
    if name not in _META_TRAINER_FACTORY:
        raise Exception('Unrecognized meta learner name: {}'.format(name))
    else:
        return _META_TRAINER_FACTORY[name](**args)

def get_available_meta_learner():
    """
    Return the list of all available meta trainer
    """
    return _META_TRAINER_FACTORY.keys()

def register_meta_trainer():
    """
    Decorator to populate _META_TRAINER_FACTORY
    """
    def decorator(cls):
        _META_TRAINER_FACTORY[cls._learner_name] = cls
        return cls
    return decorator
