import json
from statsmodels.tsa.arima_model import ARIMA

class Arima(object):
    def __init__(self, p, d, q):
        self._p = p
        self._d = d
        self._q = q

    def __call__(self, endog, exog=None):
        return ARIMA(endog=endog, order=(self.p, self.d, self.q),
                exog=exog).fit(disp=0)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        self._d = d

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q
