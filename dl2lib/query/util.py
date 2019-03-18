#import tensorflow as tf
import numpy as np
from typing import Dict, Any, List


def get_fqn(node) -> str:
    """
    Helper function that exists because textX does not correctly expose _tx_fqn as described in their
    documentation
    The function parses the value of _tx_fqn the nodes string representation.
    Ideally this should be replaced by fixing textX.
    """
    text = node.__repr__()
    if text.startswith('<textx:'):
        return text[7:].strip().split(' ')[0]
    else:
        return text.split(':')[0].replace('>', '').replace('<', '')
