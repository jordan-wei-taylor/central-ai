.. warning:: Page in development.

Gaussian Process Regression
###########################

Building on the `Bayesian Linear Regression <bayesian-linear-regression.html>`_ framework, we had to initially choose a couple of things:

+  How many basis functions should we use?
+  Where should the :math:`\mathbf{C}` locations be positioned?

The Gaussian Process (GP) answers these two problems for us! 

Recall that the general `regression <../../theories/regression.html>`_ framework is that for a dataset :math:`\mathcal{D}=\{(\mathbf{x}_1,y_1),...,(\mathbf{x}_n,y_n)\}` we want to find

.. math::

    y = f(\mathbf{x}) + \epsilon,\quad \epsilon\overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2).

What do we want our :math:`f` function to look like and are there any properties we desire? Lets make only one assumption - the function :math:`f` is smooth and we would like some error bars! If we want smoothness, we can say that if :math:`\mathbf{x}_1` and :math:`\mathbf{x}_2` are similar then the associated target values :math:`y_1` and :math:`y_2` are also similar. To help encode this we can consider a function of the norm of the difference :math:`||\mathbf{x}_1-\mathbf{x}_2||` (see :ref:`vector norm`). If we did consider this as how correlated points were, it would have the adverse effect as the norm goes to 0 when the difference goes to 0. To ensure that when the difference is small, there is high correlation, we can take the exponent of this norm. We can express this two data point example as

.. math::
    :nowrap:

    \begin{align}
        \boldsymbol{\sigma}(\mathbf{x}_1,\mathbf{x}_2) = \begin{bmatrix}\sigma_{11} & \sigma_{12}\\\sigma_{21} & \sigma_{22}\end{bmatrix} = \begin{bmatrix}\exp\big(g\left(||\mathbf{x}_1 - \mathbf{x}_1||\right)\big) & \exp\big(g\left(||\mathbf{x}_1 - \mathbf{x}_2||\right)\big)\\\exp\big(g\left(||\mathbf{x}_2 - \mathbf{x}_1||\right)\big) & \exp\big(g\left(||\mathbf{x}_2 - \mathbf{x}_2||\right)\big)\end{bmatrix},
    \end{align}

where :math:`g` is a function that we can choose. We can now encode a prior on continuous functions. One popular choice of the covariance kernel is the **squared-exponential** kernel (also known as the Gaussian or Radial Basis Function kernel) defined as

.. math::
    :nowrap:

    \begin{align}
        \Sigma_{ij} = k(\mathbf{x}_i,\mathbf{x}_j) = \sigma_s^2\exp\left(-\frac{1}{2l^2}||\mathbf{x}_i-\mathbf{x}_j||_2^2\right),
    \end{align}

where :math:`\sigma_s^2` and :math:`l` are the signal variance and lengthscale of the kernel respectively. The lengthscale modulates the relationship between the smoothness and the distance between the :math:`x` locations. Other popular choices of kernels can be found on `Kernel Cookbook <https://raw.githubusercontent.com/duvenaud/phd-thesis/master/kernels.pdf>`_ or `Wikipedia <https://en.wikipedia.org/wiki/Gaussian_process#Covariance_functions>`_.

The joint distribution between the observed dataset and a new datapoint can be now written as

.. math::
    :nowrap:

    \begin{align}
        \begin{bmatrix}\mathbf{y}\\y^{*}\end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix}m(\mathbf{X})\\m(\mathbf{x}^{*})\end{bmatrix},\begin{bmatrix}k(\mathbf{X},\mathbf{X}) & k(\mathbf{X},\mathbf{x}^{*})\\k(\mathbf{x}^{*},\mathbf{X}) & k(\mathbf{x}^{*},\mathbf{x}^{*})\end{bmatrix}\right),
    \end{align}

where :math:`m` is a mean function, :math:`\mathbf{x}^{*}` is a new observation and :math:`y^{*}` is the associated target variable. The choice of mean function is up to the user but can be as simple as the mean value of :math:`\mathbf{y}` to another model e.g. a Linear Regression model. We can solve for the posterior estimate of :math:`y^{*}` in closed form yielding

.. math::
    :nowrap:

    \begin{align}
        p(y^{*}|\mathbf{x}^{*},\mathbf{X},\mathbf{y}) &= \mathcal{N}(\mu^{*},\Sigma^{*}) \\
        \mu^{*} &= k(\mathbf{x}^{*},\mathbf{X})k(\mathbf{X},\mathbf{X})^{-1}\left(\mathbf{y} - m(\mathbf{X})\right) + m(\mathbf{x}^{*}) \\
        \Sigma^{*} &= k(\mathbf{x}^{*},\mathbf{x}^{*}) - k(\mathbf{x}^{*},\mathbf{X})k(\mathbf{X},\mathbf{X})^{-1}k(\mathbf{X},\mathbf{x}^{*})
    \end{align}

.. note::
    
    .. raw:: html
        
        <details><summary>Derivation of the (noise-free) predictive posterior</summary>

    .. math::
        :nowrap:

        \begin{align*}
            p(y^{*}|\mathbf{x}^{*},\mathbf{y},\mathbf{X}) &= \frac{p(y^{*},\mathbf{y}|\mathbf{x}^{*}\mathbf{X})}{p(\mathbf{y}|\mathbf{X})} \\
            &\propto \exp\left(-\frac{1}{2}\begin{bmatrix}\mathbf{y} - m(\mathbf{X}) \\ y^{*} - m(\mathbf{x}^{*})\end{bmatrix}^\text{T}\begin{bmatrix}\mathbf{A} & \mathbf{b}\\\mathbf{b}^\text{T} & c\end{bmatrix}\begin{bmatrix}\mathbf{y} - m(\mathbf{X}) \\ y^{*} - m(\mathbf{x}^{*})\end{bmatrix} + \frac{1}{2}\left(\mathbf{y}-m(\mathbf{X})^\text{T}\right)\left[\mathbf{K}(\mathbf{X},\mathbf{X})\right]^{-1}\left(\mathbf{y} - m(\mathbf{X})\right)\right) \\
            &\propto \exp\left(-\frac{1}{2}\left[c\left(y^{*} - m(\mathbf{x}^{*})\right)^2 + 2\mathbf{b}^\text{T}\left(\mathbf{y} - m(\mathbf{X})\right)\left(y^{*} - m(\mathbf{x}^{*})\right)\right]\right) \\
            &\propto \exp\left(-\frac{1}{2c^{-1}}\left(y^{*} - m(\mathbf{x}^{*}) + \frac{\mathbf{b}^\text{T}\left(\mathbf{y} - m(\mathbf{X})\right)}{c}\right)^2\right)\label{star}\tag{*}
        \end{align*}

    where

    .. math::

        \begin{align*}
            \begin{bmatrix}\mathbf{A} & \mathbf{b}\\\mathbf{b}^\text{T} & c\end{bmatrix} = \begin{bmatrix}\mathbf{K}(\mathbf{X},\mathbf{X}) & \mathbf{k}(\mathbf{X}, \mathbf{x}^{*})\\\mathbf{k}(\mathbf{x}^{*}, \mathbf{X}) & k(\mathbf{x}^{*},\mathbf{x}^{*})\end{bmatrix}^{-1}
        \end{align*}

    which implies 

    .. math::

        \begin{align*}
            \mathbf{K}(\mathbf{X},\mathbf{X})\mathbf{b} + c\mathbf{k}(\mathbf{X},\mathbf{x}^{*}) &= \mathbf{0}\\ 
            \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{b} + ck(\mathbf{x}^{*},\mathbf{x}^{*}) &= 1
        \end{align*}

    .. math::

        \begin{align*}
            \Rightarrow c &= \frac{1}{k(\mathbf{x}^{*},\mathbf{x}^{*}) - \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\mathbf{k}(\mathbf{X},\mathbf{x}^{*})}\\
            \Rightarrow \mathbf{b} &= \frac{-\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\mathbf{k}(\mathbf{X},\mathbf{x}^{*})}{k(\mathbf{x}^{*},\mathbf{x}^{*}) - \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\mathbf{k}(\mathbf{X},\mathbf{x}^{*})}
        \end{align*}

    Substituting into :math:`\eqref{star}` yields

    .. math::

        \begin{align*}
            p(y^{*}|\mathbf{x}^{*},\mathbf{y},\mathbf{X}) &\propto \exp\left(-\frac{1}{2}\frac{1}{k(\mathbf{x}^{*},\mathbf{x}^{*}) - \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\mathbf{k}(\mathbf{X},\mathbf{x}^{*})}\left(y^{*} - m(\mathbf{x}^{*}) - \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\left(\mathbf{y} - m(\mathbf{X})\right)\right)^2\right) \\
            &= \mathcal{N}\left(\mu^{*},\Sigma^{*}\right)\\\\
            \mu^{*} &= \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\left(\mathbf{y} - m(\mathbf{X})\right) + m(\mathbf{x}^{*})\\
            \Sigma^{*} &= k(\mathbf{x}^{*},\mathbf{x}^{*}) - \mathbf{k}(\mathbf{x}^{*},\mathbf{X})\mathbf{K}(\mathbf{X},\mathbf{X})^{-1}\mathbf{k}(\mathbf{X},\mathbf{x}^{*})
        \end{align*}

    .. raw:: html

        </details>

In practice we have :math:`y = f(\mathbf{x}) + \epsilon` with :math:`\epsilon\overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2)` so we can encorporate this independent noise yielding

.. math::
    :nowrap:

    \begin{align}
        \begin{bmatrix}\mathbf{y}\\y^{*}\end{bmatrix} &\sim \mathcal{N}\left(\begin{bmatrix}m(\mathbf{X})\\m(\mathbf{x}^{*})\end{bmatrix}, \begin{bmatrix}k(\mathbf{X,\mathbf{X}}) + \sigma^2\mathbf{I} & k(\mathbf{X},\mathbf{x}^{*}) \\ k(\mathbf{x}^{*},\mathbf{X}) & k(\mathbf{x}^{*},\mathbf{x}^{*})\end{bmatrix}\right) \label{eq:joint}\\
        p(y^{*}|\mathbf{x}^{*},\mathbf{X},\mathbf{y}) &= \mathcal{N}(\mu^{*},\Sigma^{*}) \\
        \mu^{*} &= k(\mathbf{x}^{*},\mathbf{X})\left(k(\mathbf{X},\mathbf{X}) + \sigma^2\mathbf{I}\right)^{-1}\left(\mathbf{y} - m(\mathbf{X})\right) + m(\mathbf{x}^{*})\\
        \Sigma^{*} &= k(\mathbf{x}^{*},\mathbf{x}^{*}) - k(\mathbf{x}^{*},\mathbf{X})\left(k(\mathbf{X},\mathbf{X}) + \sigma^2\mathbf{I}\right)^{-1}k(\mathbf{X},\mathbf{x}^{*})
    \end{align}

From the above we have a means of computing predictions (and quantify the uncertainty around them). By considering the first block elements in the joint distribution as defined in Eq. :math:`\eqref{eq:joint}` we have

.. math::
    :nowrap:

    \begin{align}
        \mathbf{y}|\mathbf{X},\sigma^2 \sim \mathcal{N}\left(m(\mathbf{X}),k(\mathbf{X},\mathbf{X}) + \sigma^2\mathbf{I}\right),
    \end{align}

which is also referred to as :math:`\mathcal{GP}(\mathbf{m},\mathbf{K} + \sigma^2\mathbf{I})` in the literature. Expressing the above in log probabilities we have

.. math::
    :nowrap:

    \begin{align}
        \log p(\mathbf{y}|\mathbf{X},\theta,\sigma^2) = -\frac{n}{2}\log 2\pi - \frac{1}{2}\log |k(\mathbf{X},\mathbf{X};\theta) + \sigma^2\mathbf{I}| - \frac{1}{2}\left(\mathbf{y} - m(\mathbf{X})\right)^\text{T}\left(k(\mathbf{X},\mathbf{X};\theta) + \sigma^2\mathbf{I}\right)^{-1}\left(\mathbf{y} - m(\mathbf{X})\right) \label{eq:log-likelihood},
    \end{align}

where we introduce :math:`\theta` to be the parameters of the covariance kernel. Examining the three terms from left to right, we have the normalisation constant which can be ignored from an optimisation point of view, the log determinant which measures **model complexity**, and finally the quadratic term which measures **data fit**. The GP naturally has this in-built regularisation term which penalises the model if it gets to complicated (which reduces the chance of our model overfitting). If :math:`\theta` is constrained to positive values (such as the squared exponential kernel and many other kernels) then we need to ensure it never becomes negative. One simple trick to do that is to optimise the parameters in **log** space as when we exponentiate our logarithmic values we are strictly positive.

.. note 

    Using the chain rule we can find out the gradient of a function with respect to :math:`log\theta`

    .. math::
       :nowrap:

       \begin{align*}
            \frac{\text{d}}{\text{d}\log\theta}[f(\theta)] &= \frac{\text{d}}{\text{d}z}[f(\exp(z))],\qquad (\theta = \exp(z)). \\
            &= \exp(z)f'(\exp(z)), \\
            &= \theta f'(\theta).
       \end{align*}

Let :math:`\mathbf{V} = k(\mathbf{X},\mathbf{X};\theta) + \sigma^2\mathbf{I}` then partially differentiating the log probability yields

.. math::
    :nowrap:

    \begin{align}
        \frac{\partial}{\partial \log \sigma^2}[\log p] &= -\frac{1}{2}\text{Tr}\left[\mathbf{V}^{-1}\frac{\partial}{\partial \log \sigma^2}\left[\mathbf{V}\right]\right] + \frac{1}{2}\left(\mathbf{y} - m(\mathbf{X})\right)^\text{T}\mathbf{V}^{-1}\frac{\partial}{\partial \log\sigma^2}\left[\mathbf{V}\right]\mathbf{V}^{-1}\left(\mathbf{y} - m(\mathbf{X})\right) \\
        &= -\frac{1}{2}\text{Tr}\left[\mathbf{V}^{-1}\left(\sigma^2\mathbf{I}\right)\right] + \frac{1}{2}\left(\mathbf{y} - m(\mathbf{X})\right)^\text{T}\mathbf{V}^{-1}\left(\sigma^2\mathbf{I}\right)\mathbf{V}^{-1}\left(\mathbf{y} - m(\mathbf{X})\right)\\\nonumber\\
        \frac{\partial}{\partial \log\theta}[\log p] &= -\frac{1}{2}\text{Tr}\left[\mathbf{V}^{-1}\frac{\partial}{\partial \log \theta}\left[\mathbf{V}\right]\right] + \frac{1}{2}\left(\mathbf{y} - m(\mathbf{X})\right)^\text{T}\mathbf{V}^{-1}\frac{\partial}{\partial \log\theta}\left[\mathbf{V}\right]\mathbf{V}^{-1}\left(\mathbf{y} - m(\mathbf{X})\right)
    \end{align}

Example: Load Boston
********************

.. include:: /data/load_boston.rst

.. admonition:: Python
    :class: code

    .. raw:: html

        <details><summary>Define kernel and kernel gradients</summary>

    .. code-block:: python

        from   sklearn.datasets       import load_boston
        from   scipy.spatial.distance import cdist
        from   matplotlib             import pyplot as plt
        import numpy as np

        def squared_exponential(X1, X2, l, s2):
            return s2 * np.exp(-cdist(X1, X2, 'sqeuclidean') / (2 * np.square(l)))

        def squared_exponential_gradient(X1, X2, l, s2):
            D = cdist(X1, X2, 'sqeuclidean')
            K = s2 * np.exp(-cdist(X1, X2, 'sqeuclidean') / (2 * np.square(l)))
            return [D * K, K]

        def rational_quadratic(X1, X2, l, a, s2):
            return s2 * (1 + cdist(X1, X2, 'sqeuclidean') / (2 * a * np.square(l))) ** -a

        def rational_quadratic_gradient(X1, X2, l, a, s2):
            D    = cdist(X1, X2, 'sqeuclidean')
            D2al = D / (2 * a * np.square(l))
            f    = 1 + D2al
            k    = s2 * f ** -(a + 1)
            K    = s2 * f ** -(a)
            return [s2 * 2 * D2al * k, D2al * K * np.log(f), K]

    .. raw:: html

        </details>

    .. raw:: html

        <details><summary>Gaussian Process Class</summary>

    .. code-block:: python
        :linenos:

        class GaussianProcess():
            """
            Gaussian Process Class
            
            Parameters
            ====================
                kernel          : function
                                  Covariance kernel function.
                                  
                kernel_gradient : function
                                  Computes the partial derivatives of the kernel function w.r.t. the parameters in the order they are used.
                                  
                mean_function   : function
                                  Mean function for the Gaussian Process framework.
                                  
                random_state    : int
                                  Parameter to be used in numpy.random.seed for reproducible results.
            """
            def __init__(self, kernel, kernel_gradient, mean_function = None, random_state = None):
                self.K  = kernel
                self.Kg = kernel_gradient
                self.m  = mean_function
                self._params = dict(random_state = random_state)
                
            def fit(self, X, y, alpha = 0.1, momentum = 0.1, epochs = 150):
                
                n, m   = X.shape
                
                # Generate mean function if None provided
                if self.m is None:
                    self.m = lambda X : np.ones((len(X), 1)) * y.mean()
                    
                # Shift and rescale targets
                y      = (y - self.m(X))
                scale  = self.scale = y.std()
                y     /= scale
                
                # Scale alpha
                alpha /= n
                
                np.random.seed(self._params['random_state'])
                
                # Ignore X1 and X2 but include an additional noise variance parameter
                nparams = self.K.__code__.co_argcount - 2 + 1
                
                # Initialise the log parameters and their gradients
                lparams = self.lparams = np.random.normal(scale = 0.5, size = nparams)
                gparams = np.zeros(nparams)
                
                const   = n / 2 * np.log(2 * np.pi)
                
                # Store each component of the negative log likelihood
                loss    = self.loss = np.zeros((epochs + 1, 3))
                
                for i in range(epochs):
                    K       = self.K(X, X, *np.exp(lparams[:-1]))
                    V       = K + np.eye(n) * np.exp(lparams[-1])
                    Ki      = np.linalg.inv(V)
                    Kiy     = Ki @ y
                    loss[i] = const, np.linalg.slogdet(V)[1], y.T @ Kiy
                    
                    # Update parameters (with momentum)
                    for j, grad in enumerate(self.Kg(X, X, *np.exp(lparams[:-1]))):
                        gparams[j] *= momentum
                        gparams[j] += (np.trace(Kiy.T @ grad @ Kiy) - np.trace(Ki @ grad))
                        lparams[j] += alpha * gparams[j]
                        
                    gparams[-1] *= momentum
                    gparams[-1] += np.exp(lparams[-1]) * (np.trace(Kiy.T @ Kiy) - np.trace(Ki))
                    lparams[-1] += alpha * gparams[-1]
                    
                K        = self.K(X, X, *np.exp(lparams[:-1]))
                V        = K + np.eye(n) * np.exp(lparams[-1])
                Ki       = self.Ki  = np.linalg.inv(V)
                Kiy      = self.Kiy = Ki @ y
                loss[-1] = const, np.linalg.slogdet(V)[1], y.T @ Kiy
                loss    /= n
                self.k   = lambda z : self.K(z, X, *np.exp(lparams[:-1]))
                
                return self
            
            def __call__(self, X, var = False):
                mu = self.k(X) @ self.Kiy * self.scale + self.m(X)
                if var:
                    Kxx = self.K(X, X, *np.exp(self.lparams[:-1]))
                    KxX = self.k(X)
                    cov = Kxx + KxX @ self.Ki @ KxX.T
                    var = np.diag(cov)
                    return mu, var
                return mu                                                                               

    .. raw:: html

        </details>

    .. raw:: html

        <details><summary>Load the data and create linear model and rmse function</summary>

    .. code-block:: python

        X, y = load_boston(return_X_y = True)
        y.resize(len(y), 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2021)

        w = np.linalg.lstsq(np.insert(X_train, 0, 1, 1), y_train, rcond = -1)[0]

        def linear(X):
            return np.insert(X, 0, 1, 1) @ w

        def rmse(model, X, y):
            return np.sqrt(np.mean(np.square(model(X) - y)))

    .. raw:: html

        </details>


    .. raw:: html

        <details><summary>Model training and evaluation</summary>

    .. code-block:: python

        model1 = GaussianProcess(squared_exponential, squared_exponential_gradient, mean_function = linear, random_state = 2021).fit(X_train, y_train)
        model2 = GaussianProcess(rational_quadratic , rational_quadratic_gradient , mean_function = linear, random_state = 2021).fit(X_train, y_train)
        
        # Test the trained model with both kernels against the linear case to see if there is any improvement beyond the linear model
        [rmse(model, X_test, y_test) for model in (linear, model1, model2)]
        # [4.758342005376841, 4.7583318332395335, 4.647409609017597]

    .. raw:: html

        </details>

It seems that both kernels could learn beyond the linear regression mean function (with the squared exponential kernel barely doing so). Visualising the loss over training iterations we have the following plots.

.. figure:: /_static/how-to-ml/supervised-learning/gaussian-process/kernel-comparison.png
    :align: center
    :figwidth: 70 %

    Comparison of the expected negative log likelihood of the Gaussian process with the **squared exponential** and **rational quadratic** kernels.

By comparison, it seems the rational quadratic kernel is slightly more suited to this dataset as the total log probability is lower (and continuing to decrease). Lets try to visualise the error bars on a toy problem.

Example: Noisy sin function
***************************

.. admonition:: Python
    :class: code

    .. raw:: html

        <details><summary>Data generator</summary>

    .. code-block:: python

        def generator(n, m, func, domain = (0, np.pi * 2), scale = 0.15, random_state = None):
            """
            Data Generator
            
            Parameters
            =================
                n            : int
                               Number of training samples to generate.

                m            : int
                               Number of dimensions each observation has.
                               
                func         : function
                               True function to learn.

                domain       : list, tuple
                               Domain for each dimension of our observations defining the interval [a, b].

                scale        : float
                               The standard deviation parameter for the additive Gaussian noise.

                random_state : int
                               Parameter to be used in numpy.random.seed for reproducible results.
            """
            # Set seed for X
            np.random.seed(random_state)
            
            # Training data
            xjs = [np.random.uniform(*domain, size = n) for _ in range(m)]
            X   = np.array([xj.flatten() for xj in np.meshgrid(*xjs)]).T
            t   = func(X)
            
            # Set seed for y
            np.random.seed(random_state)
            
            y   = t + np.random.normal(scale = scale, size = t.shape)
            
            # True function (at x10 resolution and 10% outside domain)
            d   = domain[0] - (domain[1] - domain[0]) * 0.1, domain[1] + (domain[1] - domain[0]) * 0.1
            xj  = [np.linspace(*d, 11 * n) for _ in range(m)]
            x   = np.array([x.flatten() for x in np.meshgrid(*xj)]).T
            t   = func(x)
            
            return X, y, x, t
        
    .. raw:: html

        </details>

    .. raw:: html

        <details><summary>Generating a training on 10 noisy sin data points</summary>

    .. code-block:: python

        X, y, x, t = generator(10, 1, np.sin, random_state = 2021)

        model1 = GaussianProcess(squared_exponential, squared_exponential_gradient, random_state = 2021).fit(X, y, alpha = 0.1, momentum = 0.8)
        model2 = GaussianProcess(rational_quadratic , rational_quadratic_gradient , random_state = 2021).fit(X, y, alpha = 0.1, momentum = 0.8)

        fig, ax = plt.subplots(1, 2, figsize = (12, 4), sharey = True)

        for i, (model, title) in enumerate(zip([model1, model2], ['Squared Exponential', 'Rational Quadratic'])):

            ax[i].plot(x, t, color = cmap.colors[0])
            ax[i].scatter(X, y, color = 'k', zorder= 3)

            mu, var = model(x, var = True)
            std     = np.sqrt(var)
            
            ax[i].plot(x, mu, color = cmap.colors[1])
            ax[i].fill_between(x.flatten(), mu.flatten() + 2 * std, mu.flatten() - 2 * std, color = cmap.colors[1], alpha = 0.5)
            ax[i].grid(ls = (0, (5, 5)))
            ax[i].set_title(title)
            
            ax[i].set_xticks(np.arange(0, 2.1, 0.5) * np.pi)
            ax[i].set_xticklabels(['$0$', r'$\rm\pi/2$', r'$\rm\pi$', r'$\rm3\pi/2$', r'$\rm2\pi$'])
            

    .. raw:: html

        </details>

.. figure:: /_static/how-to-ml/supervised-learning/gaussian-process/sin-10.png
    :align: center
    :figwidth: 70 %

    Comparison of Gaussian Process kernels on 10 noisy sin data points. Blue line is the mean with the shaded region within 2 standard deviations away from the mean.

It seems that when there is little data around, the uncertainty of our model increases! This makes intuitive sense as we are less sure about computing inference at an unfamiliar :math:`x^{*}` location. Both kernels seem to perform similarly. Lets see how that changes as we increase the number of samples from 10 to 50.

.. admonition:: Python
    :class: code

    .. raw:: html

        <details><summary>Generating a training on 50 noisy sin data points</summary>

    .. code-block:: python

        X, y, x, t = generator(50, 1, np.sin, random_state = 2021)

        model1 = GaussianProcess(squared_exponential, squared_exponential_gradient, random_state = 2021).fit(X, y, alpha = 0.1, momentum = 0.8)
        model2 = GaussianProcess(rational_quadratic , rational_quadratic_gradient , random_state = 2021).fit(X, y, alpha = 0.1, momentum = 0.8)

        fig, ax = plt.subplots(1, 2, figsize = (12, 4), sharey = True)

        for i, (model, title) in enumerate(zip([model1, model2], ['Squared Exponential', 'Rational Quadratic'])):

            ax[i].plot(x, t, color = cmap.colors[0])
            ax[i].scatter(X, y, color = 'k', zorder= 3)

            mu, var = model(x, var = True)
            std     = np.sqrt(var)
            
            ax[i].plot(x, mu, color = cmap.colors[1])
            ax[i].fill_between(x.flatten(), mu.flatten() + 2 * std, mu.flatten() - 2 * std, color = cmap.colors[1], alpha = 0.5)
            ax[i].grid(ls = (0, (5, 5)))
            ax[i].set_title(title)
            
            ax[i].set_xticks(np.arange(0, 2.1, 0.5) * np.pi)
            ax[i].set_xticklabels(['$0$', r'$\rm\pi/2$', r'$\rm\pi$', r'$\rm3\pi/2$', r'$\rm2\pi$'])
            

    .. raw:: html

        </details>

.. figure:: /_static/how-to-ml/supervised-learning/gaussian-process/sin-50.png
    :align: center
    :figwidth: 70 %

    Comparison of Gaussian Process kernels on 50 noisy sin data points. Blue line is the mean with the shaded region within 2 standard deviations away from the mean.

From the above, we can see that the uncertainty quantification is much more smooth and generally follows that of a sin function. Where we do not have any data (the far left and right regions) the uncertainty quantification increases significantly with the mean to have the tendancy to curve towards 0 (as this was assumed to be the mean function). Additionally, it seems like the rational quadratic kernel was not right for this problem as the general shape is less sin like compared to the squared exponential kernel.

.. hint::

    A periodic kernel would have been the best kernel for this problem.

Further Reading
***************

+  C.E. Rasmussen and C. K. I. Williams, *Gaussian Process for Machine Learning*, **MIT Press**, 2006, `link 1 <http://www.gaussianprocess.org/gpml/chapters/RW2.pdf>`_
+  C. Bishop, *Pattern Recognition and Machine Learning*, 2006, `link 2 <http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf#page=311>`_

