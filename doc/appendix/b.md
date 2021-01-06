.. warning:: Page in development.

################################
B: Vector and Matrix Identities
################################


.. _completing the square:

B1
==

Completing the Square
*********************

Consider we have the quadratic form where :math:`\mathbf{x}\in\mathbb{R}^{n}`, :math:`\mathbf{A}\in\mathbb{R}^{n,n}`, and :math:`\mathbf{b}\in\mathbb{R}^{n}`, then

.. math::
    :nowrap:

    \begin{align*}
        \mathbf{x}^\text{T}\mathbf{A}\mathbf{x} - 2\mathbf{x}^\text{T}\mathbf{b} &= \mathbf{x}^\text{T}\mathbf{A}\mathbf{x} - 2 \mathbf{x}^\text{T}\mathbf{b} + \mathbf{b}^\text{T}\mathbf{A}^{-1}\mathbf{b} - \mathbf{b}^\text{T}\mathbf{A}^{-1}\mathbf{b} \\
        &= (\mathbf{x} - \mathbf{A}^{-1}\mathbf{b})^\text{T}\mathbf{A}(\mathbf{x} - \mathbf{A}^{-1}\mathbf{b}) - \mathbf{b}^\text{T}\mathbf{A}^{-1}\mathbf{b}
    \end{align*}

.. _woodbury:

B2
==

Woodbury Matrix Identity
************************

Inverting a matrix :math:`\mathbf{M}` of the form :math:`\mathbf{A} + \mathbf{UCV}` where :math:`\mathbf{A}\in\mathbb{R}^{n,n}`, :math:`\mathbf{U}\in\mathbb{R}^{n,m}`, :math:`\mathbf{C}\in\mathbb{R}^{m,m}`, and :math:`\mathbf{V}\in\mathbb{R}^{m,n}` may be significantly cheaper if :math:`\mathbf{A}` and / or :math:`\mathbf{C}` are cheap to invert.

.. math::
    :nowrap:

    \begin{align*}
        \mathbf{I} &= \mathbf{I} + \mathbf{UCVA}^{-1} - \mathbf{UCVA}^{-1}, \\
        &= \mathbf{I} + \mathbf{UCVA}^{-1} - \mathbf{UC}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1}, \\
        &= \mathbf{I} + \mathbf{UCVA}^{-1} - (\mathbf{U} + \mathbf{UCVA}^{-1}\mathbf{U})(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1}, \\
        &= \mathbf{I} + \mathbf{UCVA}^{-1} - (\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1} - \mathbf{UCVA}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U}^{-1})\mathbf{VA}^{-1}, \\
        &= \mathbf{I} - \mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1} + \mathbf{UCVA}^{-1} - \mathbf{UCVA}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1}, \\
        &= \mathbf{AA}^{-1} - \mathbf{AA}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1} + \mathbf{UCV}\bigg[\bigg(\mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\bigg)\mathbf{VA}^{-1}\bigg], \\
        &= \mathbf{A}\bigg[\bigg(\mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\bigg)\mathbf{VA}^{-1}\bigg] + \mathbf{UCV}\bigg[\bigg(\mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\bigg)\mathbf{VA}^{-1}\bigg], \\
        &= (\mathbf{A} + \mathbf{UCV})\bigg[\bigg(\mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\bigg)\mathbf{VA}^{-1}\bigg],\\\\
        \Rightarrow (\mathbf{A} + \mathbf{UCV})^{-1} &= \bigg(\mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\bigg)\mathbf{VA}^{-1}.
    \end{align*}
