import tensorflow as tf

from MetaTrainer.factory import register_meta_trainer
from MetaTrainer import metaTrainer


@register_meta_trainer()
class FineTune(metaTrainer.MetaTrainer):
    """
    Class that implements a straightforward vanilla fine tuning
    """
    _valid_args = metaTrainer.MetaTrainer._valid_args
    _learner_name="FineTuner"

    def __init__(self,**kwargs):
        """
        Creation of a Vanilla fine tuner 
        """
        super(FineTune, self).__init__(**kwargs)

        self._ready=True
    
    def _validate_args(self, args):
        """
        Check that args contains everything that is needed
        """
        super(FineTune, self)._validate_args(args)
    
    def _reshape_inputs(self):
        """
        Collapse from batch of tasks to a huge batch of images
        """
        #reshape left,right,target to remove meta crap
        input_shape = self._inputs['left'].get_shape().as_list()
        out_shape_img = [input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4]]
        out_shape_gt = [input_shape[0]*input_shape[1], input_shape[2], input_shape[3], 1]
        self._inputs['left'] = tf.reshape(self._inputs['left'],out_shape_img)
        self._inputs['right'] = tf.reshape(self._inputs['right'],out_shape_img)
        self._inputs['target'] = tf.reshape(self._inputs['target'],out_shape_gt)
    
    def _build_trainer(self, args):
        """
        Create ops for forward, loss computation and backward
        """
        #directly compute the loss between model output and targets
        self._reshape_inputs()

        #forward + backward standard pass
        self._net =  self._build_forward(self._inputs['left'],self._inputs['right'],None)
        self._trainableVariables = self._net.get_trainable_variables()
        self._all_variables = self._net.get_all_variables()
        self._predictions = self._net.get_disparities()
        self._metaLoss = self._loss(self._predictions,self._inputs)
        self._metaTrain = self._optimizer.minimize(self._metaLoss)

    def _perform_train_step(self, feed_dict = None):
        _,loss,step,_ = self._session.run([self._metaTrain,self._metaLoss, self._increment_global_step,self._update_ops], feed_dict=feed_dict)
        return step,loss
    
    def _setup_summaries(self):
        self._left_summary = self._inputs['left']
        self._target_summary = self._inputs['target']
        super(FineTune, self)._setup_summaries()
    
    def _perform_summary_step(self, feed_dict = None):
        return self._session.run([self._increment_global_step,self._summary_op], feed_dict=feed_dict)