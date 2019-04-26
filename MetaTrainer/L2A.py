import tensorflow as tf 
import numpy as np 

from MetaTrainer.factory import register_meta_trainer
from MetaTrainer import metaTrainer
from Data_utils import variables, preprocessing
from Losses import loss_factory
from Nets import sharedLayers

@register_meta_trainer()
class L2A(metaTrainer.MetaTrainer):
    """
    Implementation of Learning to Adapt for Stereo
    """
    _learner_name = "L2A"
    _valid_args = [
        ("adaptationLoss", "loss for adaptation")
    ]+metaTrainer.MetaTrainer._valid_args

    def __init__(self, **kwargs):
        super(L2A, self).__init__(**kwargs)

        self._ready=True
    
    def _validate_args(self,args):
        super(L2A, self)._validate_args(args)

        if "adaptationLoss" not in args:
            print("WARNING: no adaptation loss specified, defaulting to the same loss for trainign and adaptation")
            args['adaptationLoss'] = args['loss']
        
        self._adaptationLoss = args['adaptationLoss']
    
    def _setup_inner_loop_weights(self):
        #self._w = [tf.train.exponential_decay(1.0,self._global_step,self._weight_decay_step//(self._adaptation_steps-k),0.5) for k in range(self._adaptation_steps)]+[1.0]
        self._w = [1.0] * self._adaptation_steps
    
    def _setup_gradient_accumulator(self):
        with tf.variable_scope('gradients_utils'):
            #failsafe filtering, might be never used
            self._trainableVariables = [x for x in self._trainableVariables if x.trainable==True]
            # create a copy of all trainable variables with `0` as initial values
            self._gradAccum = [tf.Variable(tf.zeros_like(tv),trainable=False) for tv in self._trainableVariables]  
            # create a op to initialize all accums vars
            self._resetAccumOp = [tv.assign(tf.zeros_like(tv)) for tv in self._gradAccum]
            # compute gradients for a batch
            self._batchGrads = tf.gradients(self._lossOp, self._trainableVariables)
            # collect the batch gradient into accumulated vars
            self._accumGradOps = [accum.assign_add(grad) for accum, grad in zip(self._gradAccum,self._batchGrads)]
            # smooth gradients
            self._gradients_to_be_applied_ops = [grad/self._metaTaskPerBatch for grad in self._gradAccum]
            # Apply accumualted gradients takes a list of couples var,grads
            self._train_op = self._optimizer.apply_gradients([(grad, var) for grad,var in zip(self._gradients_to_be_applied_ops,self._trainableVariables)])

    def _build_adaptation_loss(self, current_net, inputs):
        return self._adaptationLoss(current_net.get_disparities(), inputs)

    def _first_inner_loop_step(self, current_net):
        self._net = current_net
        self._trainableVariables =self._net.get_trainable_variables()
        self._all_variables = self._net.get_all_variables()
        self._variableState = variables.VariableState(self._session,self._all_variables)
        self._predictions = self._net.get_disparities()
        self._build_var_dict()

    def _build_trainer(self,args):
        #build model taking placeholder as input
        input_shape = self._inputs['left'].get_shape().as_list()

        self._left_input_placeholder=tf.placeholder(dtype=tf.float32,shape=input_shape[1:])
        self._right_input_placeholder=tf.placeholder(dtype=tf.float32,shape=input_shape[1:])
        self._target_placeholder=tf.placeholder(dtype=tf.float32,shape=input_shape[1:4]+[1])

        new_model_var=None
        loss_collection = []
        for i in range(self._adaptation_steps+1):
            #forward pass
            inputs = {}
            inputs['left']=tf.expand_dims(self._left_input_placeholder[i],axis=0)
            inputs['right']=tf.expand_dims(self._right_input_placeholder[i],axis=0)
            inputs['target']=tf.expand_dims(self._target_placeholder[i],axis=0)
            #perform forward
            net =  self._build_forward(inputs['left'],inputs['right'],new_model_var)

            if i!=self._adaptation_steps:
                #compute loss and gradients
                adapt_loss = self._build_adaptation_loss(net, inputs)
            
            if i==0:
                #Create variable state to handle variable updates and reset
                self._first_inner_loop_step(net)
                new_model_var = self._var_dict
            else:
                #compute eval loss
                loss_collection.append(self._loss(net.get_disparities(), inputs))

            if i!=self._adaptation_steps:
                #build updated variables 
                gradients = tf.gradients(adapt_loss, list(new_model_var.values()))
                new_model_var = self._build_updated_variables(list(self._var_dict.keys()),list(new_model_var.values()),gradients)
        
        self._setup_inner_loop_weights()
        assert(len(self._w)==len(loss_collection))
        self._lossOp = tf.reduce_sum([w*l for w,l in zip(self._w, loss_collection)])

        #create accumulator for gradients to get batch gradients
        self._setup_gradient_accumulator()


    def _perform_train_step(self, feed_dict=None):
        #read all the input data and reset gradients accumulator
        left_images,right_images,target_images, _= self._session.run([self._inputs['left'],self._inputs['right'],self._inputs['target'],self._resetAccumOp], feed_dict=feed_dict)
        
        #read variable
        var_initial_state = self._variableState.export_variables()
        
        #for all tasks
        loss=0
        for task_id in range(self._metaTaskPerBatch):
            #perform adaptation and evaluation for a single task/video sequence
            fd = {
                self._left_input_placeholder:left_images[task_id,:,:,:,:],
                self._right_input_placeholder:right_images[task_id,:,:,:,:],
                self._target_placeholder:target_images[task_id,:,:,:,:],
            }
            if feed_dict is not None:
                fd.update(feed_dict)
            _,ll=self._session.run([self._accumGradOps,self._lossOp],feed_dict=fd)
            loss+=ll

            #reset vars
            self._variableState.import_variables(var_initial_state)
        
        #apply accumulated grads to meta learn
        _,self._step_eval=self._session.run([self._train_op,self._increment_global_step],feed_dict=feed_dict)
        
        return self._step_eval,loss/self._metaTaskPerBatch
    
    def _setup_summaries(self):
        with tf.variable_scope('base_model_output'):
            self._summary_ops.append(tf.summary.image('left',self._left_input_placeholder,max_outputs=1))
            self._summary_ops.append(tf.summary.image('target_gt',preprocessing.colorize_img(self._target_placeholder,cmap='jet'),max_outputs=1))
            self._summary_ops.append(tf.summary.image('prediction',preprocessing.colorize_img(self._predictions[-1],cmap='jet'),max_outputs=1))
        self._merge_summaries()
        self._summary_ready = True
    
    def _perform_summary_step(self, feed_dict = None):
        #read one batch of data
        left_images,right_images,target_images, _ = self._session.run([self._inputs['left'],self._inputs['right'],self._inputs['target'],self._resetAccumOp], feed_dict=feed_dict)

        #for first task
        task_id=0

        #perform meta task
        fd = {
            self._left_input_placeholder:left_images[task_id,:,:,:,:],
            self._right_input_placeholder:right_images[task_id,:,:,:,:],
            self._target_placeholder:target_images[task_id,:,:,:,:]
        }
        
        if feed_dict is not None:
            fd.update(feed_dict)
        summaries,step=self._session.run([self._summary_op,self._increment_global_step],feed_dict=fd)
        return step,summaries

@register_meta_trainer()
class L2A_Wad(L2A):
    """
    Implementation of Learning to Adapt for Stereo with Confidence Weighted Adaptation
    """
    _learner_name="L2AWad"

    def __init__(self, **kwargs):
        self._reuse=False
        super(L2A_Wad, self).__init__(**kwargs)

        self._ready=True
    
    def _build_adaptation_loss(self, current_net, inputs):
        #compute adaptation loss and gradients
        reprojection_error = loss_factory.get_reprojection_loss('ssim_l1',reduced=False)(current_net.get_disparities(),inputs)[0]
        weight, self._weighting_network_vars = sharedLayers.weighting_network(reprojection_error,reuse=self._reuse,training=True)
        return tf.reduce_sum(reprojection_error*weight)
    
    def _first_inner_loop_step(self, current_net):
        self._net = current_net
        self._trainableVariables =self._net.get_trainable_variables()+[x for x in self._weighting_network_vars if x.trainable==True]
        self._all_variables = self._net.get_all_variables()
        self._variableState = variables.VariableState(self._session,self._all_variables)
        self._predictions = self._net.get_disparities()
        self._build_var_dict()
        self._reuse=True

@register_meta_trainer()
class FOL2A(L2A):
    """
    Implementation of the first order approximation of Learning to Adapt
    """
    _learner_name = "FOL2A"

    def _build_trainer(self,args):
        #build model taking placeholder as input
        input_shape = self._inputs['left'].get_shape().as_list()

        self._left_input_placeholder=tf.placeholder(dtype=tf.float32,shape=[1]+input_shape[2:])
        self._right_input_placeholder=tf.placeholder(dtype=tf.float32,shape=[1]+input_shape[2:])
        self._target_placeholder=tf.placeholder(dtype=tf.float32,shape=[1]+input_shape[2:4]+[1])

        #forward pass
        inputs = {}
        inputs['left'] = self._left_input_placeholder
        inputs['right'] = self._right_input_placeholder
        inputs['target'] = self._target_placeholder
        #perform forward
        net =  self._build_forward(inputs['left'],inputs['right'],None)

        #Create variable state to handle variable updates and reset
        self._first_inner_loop_step(net)

        #adaptation loss 
        self._adaptation_loss = self._build_adaptation_loss(net, inputs)
        self._adaptation_optimizer = tf.train.GradientDescentOptimizer(self._alpha)
        self._adaptation_train_op = self._adaptation_optimizer.minimize(self._adaptation_loss, var_list=self._trainableVariables)

        #meta evaluation loss
        self._lossOp = self._loss(net.get_disparities(), inputs)
        
        #create accumulator for gradients to get batch gradients
        self._setup_gradient_accumulator()
    
    def _perform_train_step(self, feed_dict=None):
        #read all the input data and reset gradients accumulator
        left_images,right_images,target_images, _ = self._session.run([self._inputs['left'],self._inputs['right'],self._inputs['target'],self._resetAccumOp], feed_dict=feed_dict)
        
        #read variable
        var_initial_state = self._variableState.export_variables()
        
        #for all tasks and iterations
        partial_loss = 0
        for task_id in range(self._metaTaskPerBatch):
            for it in range(self._adaptation_steps+1):
                #perform meta train
                fd = {
                    self._left_input_placeholder:np.expand_dims(left_images[task_id,it,:,:,:],axis=0),
                    self._right_input_placeholder:np.expand_dims(right_images[task_id,it,:,:,:],axis=0),
                    self._target_placeholder:np.expand_dims(target_images[task_id,it,:,:,:],axis=0)
                }
                if feed_dict is not None:
                    fd.update(feed_dict)
                if it==0:
                    # on the first frame perform only training
                    self._session.run([self._adaptation_train_op, self._update_ops],feed_dict=fd)
                elif it==(self._adaptation_steps):
                    # on the last frame perform only evaluation
                    _,lossy = self._session.run([self._accumGradOps, self._lossOp],feed_dict=fd)
                    partial_loss+=lossy
                else:
                    # on middle frame perform evaluation and adaptation
                    _,_,_,lossy = self._session.run([self._adaptation_train_op, self._update_ops,self._accumGradOps, self._lossOp],feed_dict=fd)
                    partial_loss+=lossy

            #reset vars
            self._variableState.import_variables(var_initial_state)
        
        #apply accumulated grads to meta learn
        _,step=self._session.run([self._train_op,self._increment_global_step],feed_dict=feed_dict)
        
        return step,partial_loss/self._metaTaskPerBatch
    
    def _perform_summary_step(self, feed_dict = None):
        #read one batch of data
        left_images,right_images,target_images, _ = self._session.run([self._inputs['left'],self._inputs['right'],self._inputs['target'],self._resetAccumOp], feed_dict=feed_dict)
              
        #fetch images
        fd = {
            self._left_input_placeholder:left_images[0,:1,:,:,:],
            self._right_input_placeholder:right_images[0,:1,:,:,:],
            self._target_placeholder:target_images[0,:1,:,:,:]
        }
        
        if feed_dict is not None:
            fd.update(feed_dict)

        summaries,step=self._session.run([self._summary_op,self._increment_global_step],feed_dict=fd)
        return step,summaries