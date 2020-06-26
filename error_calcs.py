def rmse(x, y):
    return (((y - x) ** 2).mean() ** 0.5)

def mape(x, y):
    return (abs(y - x) / y.mean()).mean() * 100
