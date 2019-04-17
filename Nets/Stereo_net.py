import tensorflow as tf
import abc
from collections import OrderedDict


class StereoNet(object):
    __metaclass__ = abc.ABCMeta
    """
    Meta parent class for all the convnets
    """
    #=======================Static Class Fields=============
    _valid_args = [
        ("is_training", "boolean or placeholder to specify if the network is in train or inference mode"),
        ("variable_dict", "dictionary to fetch variable from")
    ]
    _name="stereoNet"
    #=====================Static Class Methods==============

    @classmethod
    def _get_possible_args(cls):
        return cls._valid_args

    #==================PRIVATE METHODS======================
    def __init__(self, **kwargs):
        self._layers = OrderedDict()
        self._disparities = []
        self._variables_list=set()
        self._layer_to_var = {}
        print('=' * 50)
        print('Starting Creation of {}'.format(self._name))
        print('=' * 50)

        args = self._validate_args(kwargs)
        print('Args Validated, setting up graph')

        self._preprocess_inputs(args)
        print('Meta op to preprocess data created')

        self._build_network(args)
        print('Network ready')
        print('=' * 50)

    def _add_to_layers(self, name, op):
        """
        Add the layer to the network 
        Args:
            name: name of the layer that need to be addded to the network collection
            op: tensorflow op 
        """
        self._layers[name] = op

        # extract variables
        scope = '/'.join(op.name.split('/')[0:-1])
        variables_local = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
        variables_global =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        variables = variables_local+variables_global
        self._layer_to_var[name] = variables
        self._variables_list.update(variables)

    def _get_layer_as_input(self, name):
        if name in self._layers:
            return self._layers[name]
        else:
            raise Exception('Trying to fetch an unknown layer!')

    def __str__(self):
        """to string method"""
        ss = ""
        for k, l in self._layers.items():
            if l in self._disparities:
                ss += "Prediction Layer {}: {}\n".format(k, str(l.shape))
            else:
                ss += "Layer {}: {}\n".format(k, str(l.shape))
        return ss

    def __repr__(self):
        """to string method"""
        return self.__str__()

    def __getitem__(self, key):
        """
        Returns a layer by name
        """
        return self._layers[key]

    #========================ABSTRACT METHODs============================
    @abc.abstractmethod
    def _preprocess_inputs(self, args):
        """
        Abstract method to create metaop that preprocess data before feeding them in the network
        """

    @abc.abstractmethod
    def _build_network(self, args):
        """
        Should build the elaboration graph
        """
        pass

    @abc.abstractmethod
    def _validate_args(self, args):
        """
        Should validate the argument and add default values
        """
        portion_options = ['BEGIN', 'END']
        # Check common args
        if 'is_training' not in args:
            print('WARNING: flag for trainign not setted, using default False')
            args['is_training']=False
        if 'variable_collection' not in args:
            print('WARNING: no variable collection specified using the default one')
            args['variable_collection']=None

        # save args value
        self._variable_collection = args['variable_collection']
        self._isTraining=args['is_training']

    #==============================PUBLIC METHODS==================================

    def get_all_layers(self):
        """
        Returns all network layers
        """
        return self._layers
    
    def get_layers_names(self):
        """
        Returns all layers name
        """
        return self._layers.keys()

    def get_disparities(self):
        """
        Return all the disparity predicted with increasing resolution
        """
        return self._disparities

    def get_variables(self, layer_name):
        """
        Returns the colelction of variables associated to layer_name
        Args:
        layer_name: name of the layer for which we want to access variables
        """
        if layer_name in self._layers and layer_name not in self._layer_to_var:
            return []
        else:
            return self._layer_to_var[layer_name]

    def get_all_variables(self):
        """
        Return a list with all the variables defined inside the graph
        """
        return list(self._variables_list)

    def get_trainable_variables(self):
        """
        Return a list with all the variable with trainable = True
        """
        return [x for x in list(self._variables_list) if x.trainable]
