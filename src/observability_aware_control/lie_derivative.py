"""
Lie Derivative Module.

Copyright © 2024 H S Helson Go and Ching Lok Chong

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import functools

import jax


def _lfh_impl(fun, vector_field, x, u, *args, **kwargs):
    _, f_jvp = jax.linearize(lambda x_: fun(x_, u, *args, **kwargs), x)
    return f_jvp(vector_field(x, u))


def lie_derivative(fun, vector_field, order):
    """Compute the Lie Derivative of a function along a vector field."""
    # Zeroth-order Lie Derivative
    lfh = fun

    # Implement the recurrence relationship for higher order lie derivatives
    for _ in range(order + 1):
        yield lfh
        lfh = functools.partial(_lfh_impl, lfh, vector_field)
