import functools
import operator

import jax
import jax.numpy.linalg as jla


class SimpleEKF:
    def __init__(
        self,
        fcn,
        hfcn,
        in_cov,
        obs_cov,
        resid_fcn=operator.sub,
        corr_fcn=operator.add,
    ):
        self.fcn = fcn
        self.hfcn = hfcn
        self._fjac = jax.jacobian(fcn, argnums=(0, 1))
        self._hjac = jax.jacobian(hfcn)
        self._in_cov = in_cov
        self._obs_cov = obs_cov
        self._resid_fcn = resid_fcn
        self._corr_fcn = corr_fcn

    @property
    def in_cov(self):
        return self._in_cov

    @property
    def obs_cov(self):
        return self._obs_cov

    @functools.partial(jax.jit, static_argnums=[0], donate_argnums=[1, 2])
    def predict(self, x_op, kf_cov, u, dt):
        fjac, gjac = self._fjac(x_op, u, dt)
        kf_cov = jla.multi_dot((fjac, kf_cov, fjac.T)) + jla.multi_dot((
            gjac,
            self.in_cov,
            gjac.T,
        ))
        x_op = self.fcn(x_op, u, dt)
        return x_op, kf_cov

    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def update(self, x_op, kf_cov, y, args=()):
        if y.size == 0:
            return x_op, kf_cov

        hjac = self._hjac(x_op, *args)

        hx = self.hfcn(x_op, *args)

        resid_cov = hjac @ kf_cov @ hjac.T + self.obs_cov
        kf_gain = jla.solve(resid_cov.T, hjac @ kf_cov).T
        kf_cov -= (
            kf_gain @ hjac @ kf_cov
            + kf_cov @ hjac.T @ kf_gain.T
            - kf_gain @ resid_cov @ kf_gain.T
        )

        z = self._resid_fcn(y, hx).ravel()
        x_op = self._corr_fcn(x_op, kf_gain @ z)
        return x_op, kf_cov
