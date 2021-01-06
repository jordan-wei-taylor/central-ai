.. warning:: Page in development.

################################
C: Vector and Matrix Calculus
################################

It is assumed that all vectors are of size :math:`n` and similarly all matrices are of size :math:`n \times n`.

.. _differential of an inverse:

C1
==

:math:`\frac{\partial}{\partial \theta}\left[\mathbf{A}^{-1}(\theta)\right] = -\mathbf{A}^{-1}\frac{\partial}{\partial \theta}\left[\mathbf{A}\right]\mathbf{A}^{-1}`
*********************************************************************************************************************************************************************

.. math::
    :nowrap:

    \begin{align*}
        \mathbf{I} &= \mathbf{A}\mathbf{A}^{-1} \\
        \partial[\mathbf{I}] &= \partial\left[\mathbf{A}\right]\mathbf{A}^{-1} + \mathbf{A}\partial\left[\mathbf{A}^{-1}\right], \\
        \mathbf{0} &= \partial\left[\mathbf{A}\right]\mathbf{A}^{-1} + \mathbf{A}\partial\left[\mathbf{A}^{-1}\right], \\
        \Rightarrow \mathbf{A}\partial\left[\mathbf{A}^{-1}\right] &= -\partial\left[\mathbf{A}\right]\mathbf{A}^{-1}, \\
        \partial\left[\mathbf{A}^{-1}\right] &= -\mathbf{A}^{-1}\partial\left[\mathbf{A}\right]\mathbf{A}^{-1}.
    \end{align*}


.. _differential of a det:

C2
==

:math:`\frac{\partial}{\partial \theta}\left[|\mathbf{A}(\theta)|\right] = |\mathbf{A}|\text{Tr}\left[\mathbf{A}^{-1}\frac{\partial}{\partial \theta}\left[\mathbf{A}\right]\right]`
************************************************************************************************************************************************************************************

Starting with :math:`|\mathbf{I}| = 1`, observe how this value changes if we add :math:`h\mathbf{A}` i.e.

.. math::

    \begin{align*}
        |\mathbf{I} + h\mathbf{A}| 
        &= \left|\begin{bmatrix}
            1 + ha_{11} & ha_{12}     & \ldots & ha_{1n} \\ 
            ha_{21}     & 1 + ha_{22} &        & ha_{2n} \\
            \vdots      &             & \ddots & \vdots  \\
            ha_{n1}     & \ldots      & \ldots & 1 + ha_{nn}
        \end{bmatrix}\right|, \\
        &= \prod_{i=1}^n (1 + ha_{ii}) + \mathcal{O}(h^2), \\
        &= 1 + h\sum_{i=1}^n a_{ii} + \mathcal{O}(h^2), \\
        &= 1 + h\text{Tr}[A] + \mathcal{O}(h^2).
    \end{align*}

By first principles, we can then show that the differential of :math:`|\mathbf{I}|` with respect to :math:`\mathbf{A}` is

.. math::

    \begin{align*}
        \nabla_{\mathbf{A}} |\mathbf{I}| &= \lim_{h\rightarrow 0}\frac{|\mathbf{I} + h\mathbf{A}| - |\mathbf{I}|}{h}, \\
        &= \lim_{h\rightarrow 0} \frac{1 + h\text{Tr}[\mathbf{A}] + \mathcal{O}(h^2) - 1}{h}, \\
        &= \lim_{h\rightarrow 0} \text{Tr}[\mathbf{A}] + \mathcal{O}(h), \\
        &= \text{Tr}[\mathbf{A}].
    \end{align*}

If :math:`\mathbf{A} = \mathbf{A}(\theta)` then by chain rule,

.. math::

    \frac{\partial |\mathbf{I}|}{\partial \theta} = \text{Tr}\left[\frac{\partial \mathbf{A}}{\partial \theta}\right].

So far we have shown how a determinant changes if we had the identity matrix. Lets define :math:`\mathbf{B}\in\mathbb{R}^{n,n}`, then using the fact that :math:`|\mathbf{B}\mathbf{A}| = |\mathbf{B}||\mathbf{A}|`, and the chain rule, we have

.. math::

    \begin{align*}
        \partial\left[|\mathbf{B}|\right] &= \partial\left[|\mathbf{AA}^{-1}\mathbf{B}|\right], \\
        &= |\mathbf{A}|\partial\left[|\mathbf{A}^{-1}\mathbf{B}|\right], \\
        &= |\mathbf{A}|\text{Tr}\left[\mathbf{A}^{-1}\partial\left[\mathbf{B}\right]\right]. \quad \text{(conditioned that } \mathbf{A}^{-1}\mathbf{B} = \mathbf{I} \text{ so we can use the above)}
    \end{align*}

Substituting :math:`\mathbf{B} = \mathbf{A}` we finally have

.. math::

    \partial\left[|\mathbf{A}|\right] = |\mathbf{A}|\text{Tr}\left[\mathbf{A}^{-1}\partial\left[\mathbf{A}\right]\right]

.. _differential of a logdet:

C3
==

:math:`\frac{\partial}{\partial \theta}\left[\log |\mathbf{A}(\theta)|\right] = \text{Tr}\left[\mathbf{A}^{-1}\partial\left[\mathbf{A}\right]\right]`
*****************************************************************************************************************************************************

Using the chain rule we know that :math:`\partial\left[f(g(\theta))\right] = g'(\theta)f'(g(\theta))`. Let :math:`f` be the log function and :math:`g(\theta) = |\mathbf{A}|` then by using the chain rule we have

.. math::

    \partial\left[\log |\mathbf{A}|\right] = \frac{1}{|\mathbf{A}|} |\mathbf{A}| \text{Tr}\left[\mathbf{A}^{-1}\partial\left[\mathbf{A}\right]\right] = \text{Tr}\left[\mathbf{A}^{-1}\partial\left[\mathbf{A}\right]\right].






