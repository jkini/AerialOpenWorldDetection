import numpy as np
import scipy.optimize as sciopt

def hungarian_matcher(cost):
  """Computes Hungarian Matching given a single cost matrix.

  Relevant DETR code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

  Args:
    cost: Matching cost matrix of shape [N, M].

  Returns:
    Array of shape [min(N, M), 2] where each row contains a matched pair of
    indices into the rows (N) and columns (M) of the cost matrix.
  """
  # Matrix is transposed to maintain the convention of other matchers:
  col_ind, row_ind = sciopt.linear_sum_assignment(cost.T)
  return np.stack([row_ind, col_ind])


if __name__ == "__main__":
  hungarian_matcher(np.ones((10, 10))*np.nan)
