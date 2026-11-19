"""Use with a CUDA build of tagged XGBoost that includes 05-shap-device.patch."""
import ctypes
import numpy as np
from xgboost.core import _LIB, _check_call, c_bst_ulong

def exact_gpu_shap_abs_sum(booster, matrix):
    out = np.empty(matrix.num_col(), dtype=np.float64)
    fn = _LIB.XGBoosterExactShapAbsSumCuda
    fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p, c_bst_ulong, ctypes.POINTER(ctypes.c_double))
    fn.restype = ctypes.c_int
    _check_call(fn(booster.handle, matrix.handle, out.size,
                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    return out
