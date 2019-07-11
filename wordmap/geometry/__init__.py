def rotate(X, theta, axis):
  '''
  Rotate multidimensional array `X` `theta` degrees
  around axis `axis`

  @args:
    {numpy.ndarray} X: a numpy array with shape (n, 3)
    {float} theta: the amount to rotate `X` around `axis` in radians
    {str} axis: the axis on which to rotate X: 'x', 'y', or 'z'
  @returns:
    {numpy.ndarray} X after the rotation has been applied

  Example:
    import numpy as np
    from wordmap.geometry import rotate
    df = np.random.rand(100,3)
    rotated = rotate(df, np.pi/2, 'y')
  '''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x':
    return np.dot(X, np.array([
      [1.,  0,  0],
      [0 ,  c, -s],
      [0 ,  s,  c]
    ]))
  elif axis == 'y':
    return np.dot(X, np.array([
      [c,  0,  -s],
      [0,  1,   0],
      [s,  0,   c]
    ]))
  elif axis == 'z':
    return np.dot(X, np.array([
      [c, -s,  0 ],
      [s,  c,  0 ],
      [0,  0,  1.],
    ]))
