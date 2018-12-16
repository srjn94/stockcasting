"""Define the model."""

import tensorflow as tf

def average_block(out, name):
    with tf.variable_scope(name):
        num_nonzero = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(out, axis=-1), 0), tf.float32))
        out = tf.reduce_sum(out, axis=-1)
        out = tf.divide(out, num_nonzero)
    return out

def recurrent_block(out, block, name):
    with tf.variable_scope(name):
        cell_fw = tf.nn.rnn_cell.GRUCell(**block["cell_fw"])
        cell_bw = tf.nn.rnn_cell.GRUCell(**block["cell_bw"])
        out, _, _ = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, out, **block["rnn"], dtype=tf.float32)
    return out

def dense_block(out, block, name):
    with tf.variable_scope(name):
        out = tf.layers.dense(out, **block["dense"])
        if "bn" in block:
            out = tf.layers.batch_normalization(out, **block["bn"])
        if "dropout" in block:
            out = tf.layers.dropout(out, **block["dropout"])
    return out

def output_block(out, block, name):
    with tf.variable_scope(name):
        out = tf.layers.dense(out, **block["logits"])
    return out

def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = inputs['corpora']
    print(out)
    if params.model_version == "mlp":
        out = average_block(out, "average")
        print(out)
        out = tf.contrib.layers.flatten(out)
        print(out)
        for i, block in enumerate(params.dense_blocks):
            out = dense_block(out, block, f"dense_{i}")
            print(out)
        out = output_block(out, params.output_block, "output")

    elif params.model_version == "rec":
        out = average_block(out, "average")
        for i, block in enumerate(params.recurrent_blocks):
            out = recurrent_block(out, block, f"recurrent_{i}")
        for i, block in enumerate(params.dense_blocks):
            out = dense_block(out, block, f"dense_{i}")
        out = output_block(out, params.output_block, "output")

    elif params.model_version == 'att':
        raise NotImplementedError()

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return out

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['stock']
    print(labels)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)
        predictions = tf.argmax(logits, -1)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    print(logits)
    print(labels)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(losses)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
