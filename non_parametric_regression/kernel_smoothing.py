def smooth(x, y, p, metric, kernel, h):
    _sum = 0.
    _sum1 = 0.
    for i in range(len(x)):
        ker = kernel(metric(p, x[i]) / h)
        if abs(ker) > 1.:
            ker = 0.
        _sum += ker * y[i]
        _sum1 += ker
    return _sum / _sum1
