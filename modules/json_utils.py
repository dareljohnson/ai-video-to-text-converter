"""
JSON utilities for handling numpy types.

Authors: Bai Blyden, Darel Johnson
"""

import json
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def json_dump(obj, fp, **kwargs):
    """Dump to JSON handling numpy types."""
    return json.dump(obj, fp, cls=NumpyJSONEncoder, **kwargs)

def json_dumps(obj, **kwargs):
    """Dumps to JSON string handling numpy types."""
    return json.dumps(obj, cls=NumpyJSONEncoder, **kwargs)
