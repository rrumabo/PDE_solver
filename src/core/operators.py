# Works for sparse/dense matrices (supports @) or callables (L_op(u))
def apply_op(L_op, u):
    return (L_op @ u) if hasattr(L_op, "__matmul__") else L_op(u)