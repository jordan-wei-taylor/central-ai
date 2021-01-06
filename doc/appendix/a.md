.. warning:: Page in development.

###########################
A: Linear Algebra
###########################

.. _inner product:

A1
==

Inner Product
*************

Consider we have a vectors :math:`\mathbf{x}\in\mathbb{R}^n` and :math:`\mathbf{y}\in\mathbb{R}^n` then we define the inner product, :math:`\langle\cdot,\cdot\rangle: \mathbb{R}^n,\ \mathbb{R}^n \rightarrow \mathbb{R}` as

.. math::
    
    \langle\mathbf{x},\mathbf{y}\rangle = \mathbf{x}^\text{T}\mathbf{y} = \mathbf{x} \cdot \mathbf{y}.

Defining first :math:`a\in\mathbb{R}`, the inner product has the following properties:
    
    1. Linearity
        

        .. math::
            
            \begin{align*}
                \langle a \mathbf{x},\mathbf{y}\rangle &= \langle\mathbf{x},a\mathbf{y}\rangle = a \langle\mathbf{x},\mathbf{y}\rangle\\
                \langle \mathbf{x} + \mathbf{y}, \mathbf{z}\rangle &= \langle\mathbf{x},\mathbf{z}\rangle + \langle\mathbf{y},\mathbf{z}\rangle
            \end{align*}


    2. Conjuagacy
        
        .. math::
        
            \langle\mathbf{x},\mathbf{y}\rangle = \langle\mathbf{y},\mathbf{x}\rangle

    3. Semi-positive definite

        .. math::

            \langle\mathbf{x},\mathbf{x}\rangle \ge 0\quad

       with equality if and only if  :math:`\mathbf{x} = \mathbf{0}`

.. _vector norm:

A2
==

Vector Norm
***********

Consider we have a vector :math:`\mathbf{z}\in\mathbb{R}^n` then the vector norm is defined by


.. math::

    ||\mathbf{z}||_p = \bigg[\sum_{i = 1}^n |z_i|^p\bigg]^{1/p}

The 2-norm is induced by the square root of its inner product :math:`\sqrt{\langle\mathbf{z},\mathbf{z}\rangle}` and is often referred to as the **Euclidean** distance. Other popular settings of :math:`p` are

.. math::

    \begin{align*}
        ||\mathbf{z}||_1 &= \sum_{i = 1}^n |z_i|\\\\
        ||\mathbf{z}||_\infty &= \max_{i = 1,...,n} |z_i|
    \end{align*}

The common setting if :math:`p` is not stated is 2 i.e. :math:`||\mathbf{x}|| = ||\mathbf{x}||_2`.

.. _cauchy-schwarz:

A3
==

Cauchy-Schwarz Inequality
*************************

The Cauchy-Schwarz inequality state that for :math:`\mathbf{x}\in\mathbb{R}^n`, :math:`\mathbf{y}\in\mathbb{R}^n`, it is true that


.. math::

    |\langle\mathbf{x}, \mathbf{y}\rangle| \le ||\mathbf{x}||||\mathbf{y}||

Proof.

Let :math:`\mathbf{z} = \mathbf{x} - \frac{\langle\mathbf{x},\mathbf{y}\rangle}{\langle\mathbf{y},\mathbf{y}\rangle}\mathbf{y}` then by linearity of the inner product, we have

.. math::

    \begin{align*}
        \langle \mathbf{z},\mathbf{y}\rangle &= \bigg\langle\mathbf{x} - \frac{\langle\mathbf{x},\mathbf{y}\rangle}{\langle\mathbf{y},\mathbf{y}\rangle}\mathbf{y},\mathbf{y}\bigg\rangle, \\
        &= \langle\mathbf{x},\mathbf{y}\rangle - \frac{\langle\mathbf{x},\mathbf{y}\rangle}{\langle\mathbf{y},\mathbf{y}\rangle}\langle\mathbf{y},\mathbf{y}\rangle = 0,
    \end{align*}

which implies that :math:`\mathbf{z}` is orthogonal to :math:`\mathbf{y}`. We can then apply Pythagoras' theorem to

.. math::

    \begin{align*}
        \mathbf{x} &= \frac{\langle\mathbf{x},\mathbf{y}\rangle}{\langle\mathbf{y},\mathbf{y}\rangle}\mathbf{y} + \mathbf{z}, \\
        \Rightarrow ||\mathbf{x}||^2 &= \bigg|\frac{\langle\mathbf{x},\mathbf{y}\rangle}{\langle\mathbf{y},\mathbf{y}\rangle}\bigg|^2||\mathbf{y}||^2 + ||\mathbf{z}||^2, \\
        &= \frac{|\langle\mathbf{x},\mathbf{y}\rangle|^2}{\big(||\mathbf{y}||^2\big)^2}||\mathbf{y}||^2 + ||\mathbf{z}||^2, \\
        &= \frac{|\langle\mathbf{x},\mathbf{y}\rangle|^2}{||\mathbf{y}||^2} + ||\mathbf{z}||^2, \\
        &\ge \frac{|\langle\mathbf{x},\mathbf{y}\rangle|^2}{||\mathbf{y}||^2}, \\
        \Rightarrow ||\mathbf{x}||^2||\mathbf{y}||^2 &\ge |\langle\mathbf{x},\mathbf{y}\rangle|^2, \\
        ||\mathbf{x}||||\mathbf{y}|| &\ge |\langle\mathbf{x},\mathbf{y}\rangle|.
    \end{align*}

.. _triangle inequality:

A4
==

Triangle Inequality
*******************

The triangle inequality states that for :math:`\mathbf{x}\in\mathbb{R}^n` and :math:`\mathbf{y}\in\mathbb{R}^n` we have

.. math::

    ||\mathbf{x} + \mathbf{y}|| \le ||\mathbf{x}|| + \mathbf{y}.

Proof.

.. math::

    \begin{align*}
        ||\mathbf{x} + \mathbf{y}||^2 &= ||\mathbf{x}||^2 + 2\langle\mathbf{x},\mathbf{y}\rangle + ||\mathbf{y}||^2, \\
        &\le ||\mathbf{x}||^2 + 2||\mathbf{x}||||\mathbf{y}||| + ||\mathbf{y}||^2,\quad\text{(by Cauchy-Schwarz)} \\
        &= \bigg(||\mathbf{x}|| + ||\mathbf{y}||\bigg)^2, \\
        \Rightarrow ||\mathbf{x} + \mathbf{y}|| &\le ||\mathbf{x}|| + ||\mathbf{y}||.
    \end{align*}

.. _matrix norm:

A5
==

Matrix Norm
***********

Consider the matrix :math:`\mathbf{A}\in\mathbb{R}^{n,m}` then the element-wise matrix norm is defined as

.. math::

    ||\mathbf{A}||_p = \bigg[\sum_{i=1}^n\sum_{j=1}^m|a_{ij}|^p\bigg]^{1/p}.

Stating the most common use cases of :math:`p`, we have

.. math::

    \begin{align*}
        ||\mathbf{A}||_1 &= \sum_{i=1}^n\sum_{j=1}^m |a_{ij}|, \\
        ||\mathbf{A}||_\infty &= \max\big\{|a_{ij},\ 1\le i \le n,\ 1\le j\le m |\big\}, \\
        ||\mathbf{A}||_2 &= ||\mathbf{A}||_\mathcal{F} = \sqrt{||\mathbf{A}||_2^2}, \\
        &= \sqrt{||\mathbf{A}^\text{T}\mathbf{A}||_2}, \\
        &= \sqrt{||\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\text{T}||_2}, \\
        &= \sqrt{||\boldsymbol{\Lambda}||_2}, \\
        &= \sqrt{\sum_{j=1}^m \lambda_j^2}
    \end{align*}

where :math:`||\cdot||_\mathcal{F}` is known as the **Frobenius** norm, :math:`\mathbf{A}^\text{T}\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\text{T}` is known as the eigen-decomposition with :math:`\mathbf{V}=[\mathbf{v}_1,...,\mathbf{v}_m]` and :math:`\boldsymbol{\Lambda} = \text{diag}(\lambda_1,...,\lambda_m)` are known as eigenvectors and eigenvalues respectively where :math:`\lambda_1 \ge \lambda_2\ge...\ge\lambda_m`.

Alternative to the element-wise matrix norms, there is also the induced (or operator) norm. Defining :math:`\mathbf{x}\in\mathbb{R}^m` we have

.. math::

    \begin{align*}
        ||\mathbf{A}|| &= \sup_{||\mathbf{x}|| = 1}\big\{||\mathbf{Ax}||\big\} \\
        &= \sup_{||\mathbf{x}|| = 1}\big\{\sqrt{||\mathbf{Ax}||^2}\big\} \\
        &= \sup_{||\mathbf{x}|| = 1}\big\{\sqrt{\mathbf{x}^\text{T}\mathbf{A}^\text{T}\mathbf{Ax}}\big\} \\
        &= \sup_{||\mathbf{x}|| = 1}\big\{\sqrt{\mathbf{x}^\text{T}\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\text{T}\mathbf{x}}\big\} \\
        &= \sup_{||\mathbf{x}|| = 1}\bigg\{\sqrt{\mathbf{y}^\text{T}\boldsymbol{\Lambda}\mathbf{y}}\bigg\} \\
        &= \sup_{||\mathbf{y}|| = 1}\bigg[\sum_{i=1}^m y_i^2 \lambda_i\bigg]^{1/2} \\
        &= \sqrt{\lambda_1}
    \end{align*}

where we again, use the eigen-decomposition, and let :math:`\mathbf{y} = \mathbf{V}^\text{T}\mathbf{x}`. We can observe that if :math:`y_1 = 1` and :math:`y_i = 0` if :math:`i > 1`, then we obtain the maximum (with the constraint that :math:`||\mathbf{y}|| = 1`). 
