from tensorflow.keras import backend as K


def positives(y_true, y_pred):
    return K.sum(y_true)


def negatives(y_true, y_pred):
    return K.sum(1 - y_true)


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def true_negatives(y_true, y_pred):
    y_pred_neg = K.round(K.clip(1 - y_pred, 0, 1))
    y_neg = K.round(K.clip(1 - y_true, 0, 1))
    return K.sum(K.round(K.clip(y_neg * y_pred_neg, 0, 1)))


def false_negatives(y_true, y_pred):
    y_pred_neg = K.round(K.clip(1 - y_pred, 0, 1))
    return K.sum(y_true * y_pred_neg)


def false_positives(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_neg = K.round(K.clip(1 - y_true, 0, 1))
    fp = K.sum(y_neg * y_pred_pos)
    return fp
  

def precision_positives(y_true, y_pred):
    """Number of correct positives out all predicted positives."""
    true_positives_computed = true_positives(y_true, y_pred)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives_computed / (predicted_positives + K.epsilon())
    return precision


def precision_negatives(y_true, y_pred):
    """Number of correct negatives out all predicted negatives."""
    true_negatives_computed = true_negatives(y_true, y_pred)
    predicted_negatives = K.sum(K.round(K.clip(1 - y_pred, 0, 1)))
    precision = true_negatives_computed / (predicted_negatives + K.epsilon())
    return precision


def recall_positives(y_true, y_pred):
    """Also called sensitivity or true positive rate."""
    true_positives_computed = true_positives(y_true, y_pred)
    possible_positives = positives(y_true, y_pred)
    return true_positives_computed / (possible_positives + K.epsilon())


def recall_negatives(y_true, y_pred):
    """Also called specificity of true negative rate."""
    true_negatives_computed = true_negatives(y_true, y_pred)
    possible_negatives = negatives(y_true, y_pred)
    return true_negatives_computed / (possible_negatives + K.epsilon())
