from typing import Iterable, Optional, Dict, Any, Generator

SOLVER_SUPPORT_INFO: Dict[str, Dict[str, Any]] = {
    "lbfgs": {
        "penalties": ["l2", None],
        "dual": False,
    },
    "liblinear": {
        "penalties": ["l1", "l2"],
        "dual": True,
    },
    "newton-cg": {
        "penalties": ["l2", None],
        "dual": False,
    },
    "newton-cholesky": {
        "penalties": ["l2", None],
        "dual": False,
    },
    "sag": {
        "penalties": ["l2", None],
        "dual": False,
    },
    "saga": {
        "penalties": ["elasticnet", "l1", "l2", None],
        "dual": False,
    },
}

def get_valid_solver_info(solver: str) -> Dict[str, Any]:
    solvers = SOLVER_SUPPORT_INFO.keys()
    if solver not in solvers:
        raise ValueError(f"Invalid solver, expects one of {list(solvers)}, but got {solver} instead")
    return SOLVER_SUPPORT_INFO[solver]
    
