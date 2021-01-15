.. warning:: Page in development.

Neural Networks
##########################

The (artifical) neural network builds on the `logistic regression <linear-regression.html>`_ framework with use of mappings to *hidden* latent spaces. To everyone's surprise, the concept of neural networks dates back to (McCullcoch & Pitts, 1943)\ [1]_. Originally it worked by using a system of nested logic gates but without the availablility of the modern computational power. It was primarily due to this lack of computational power that neural networks are only trending recently with the modern computational advances. 

The simplest form of Neural Network is known as the MultiLayered Perceptron\ [2]_ which builds on the concepts of *Perceptrons* as the name suggests. It is also known as a **fully connected** network with **dense** connections.

The Perceptron
==============

The perceptron can be thought of as a binary classification model and works by taking a linear sum and evaluating a step function.

.. figure:: /_static/how-to-ml/supervised-learning/neural-networks/perceptron.png
    :align: center
    :figwidth: 50 %
    
    Illustration of perceptron with three features. The output, :math:`y_1` is a step function on :math:`\sum_{i=1}^3 x_iw_i + b_0`.

The step function the Perceptron uses is a piecewise function of the form

.. math::
    :nowrap:

    \begin{align}
        y_1 = \phi_\text{step}(z) = \begin{cases}1, & \text{if } z = \sum_{i=1}^3 x_iw_i + b_0 \ge c,\\0, & \text{otherwise},\end{cases}
    \end{align}

where :math:`c` can be set to :math:`0` for simplicity. Examining the step function yields a function that is difficult to train i.e. compute gradients. For the binary classification case, it is common to use the sigmoid function which is naturally more smooth and gives a probabilistic interpretation. The sigmoid function is defined as

.. math::
    :nowrap:

    \begin{align}
        \phi_\text{sigmoid}(z) = \frac{1}{1 + \exp(-z)},
    \end{align}

where the sigmoid function maps to the interval :math:`(0, 1)` rather than the set :math:`\{0, 1\}` which the step function yields.

.. figure:: /_static/how-to-ml/supervised-learning/neural-networks/step-sigmoid.png
    :align: center
    :figwidth: 70 %

    Comparison on the step and sigmoid functions.
    
It should be noted that there are other functions to consider apart from the sigmoid function. For a more complete list refer to `Wikipedia <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>`_.

It seems what we have done so far is just badly define a logistic model....so whats the point? We can repeat this step multiple times!

Multi-Layered Perceptron
========================

Now that we have defined the Perceptron, the Multi-Layered Perceptron (MLP) is simply stacking Perceptrons on Perceptrons.

Training an MLP may seem challenging but thanks to [Rumelhart & Hinton & Williams, 1986]\ [3]_, we have the **backpropagation** algorithm which is just the application of the *chain rule* in context of layered learning. Consider the `regression <../theories/regression.html>`_ case where it is of interest to minimise the *mean squared error*

.. math::
    :nowrap:
    
    \begin{align}
        \mathcal{L} = \frac{1}{2}||\mathbf{y} - \hat{\mathbf{y}}||_2^2.\label{eq:loss}
    \end{align}

We can recursively define an MLP by

.. math::
    :nowrap:
    
    \begin{align}
        \mathbf{H}_{l + 1} &= \boldsymbol{\phi}_l\left(\mathbf{H}_l\mathbf{W}_l + \mathbf{b}_l\right),\quad\forall\ l\in\{0,1,...,L - 1\}\label{eq:recursive}\\\nonumber\\
        \mathbf{H}_0 &= \mathbf{X},\\
        \hat{\mathbf{y}} &= \mathbf{H}_L,
    \end{align}

where the network is made up of :math:`L - 1` **hidden** layers where the final layer is our predictor for :math:`\mathbf{y}` conditioned on the input data :math:`\mathbf{X}` and model parameters to learn. Each layer can have a different *activation* function :math:`\boldsymbol{\phi}` but is most commonly the identity function in the case of regression i.e. :math:`\boldsymbol{\phi}_{L-1} : x \rightarrow x`. By differentiating Eq. :math:`\eqref{eq:loss}`, we naturally define

.. math::
    :nowrap:
    
    \begin{align}
        \nabla_{\hat{\mathbf{y}}}\mathcal{L} = \nabla_{\mathbf{H}_{L}}\mathcal{L} = \mathbf{e} = \hat{\mathbf{y}} - \mathbf{y},
    \end{align}

to be the residual errors we want to minimise. It can be shown that the gradients of our network can be computed as 

.. math::
    :nowrap:
    
    \begin{align}
        \nabla_{\mathbf{W}_{L - k - 1}} \mathcal{L} &= \mathbf{H}_{L - k - 1}^\text{T} \cdot \boldsymbol{\phi}_{L-k-1}' \cdot \prod_{l = L - k}^{L - 1}\left[\boldsymbol{\phi}_{l}' \cdot \mathbf{W}_{l}^\text{T}\right] \cdot \mathbf{e}, & \text{for } k\in\{0,1,...,L\} \\
        \nabla_{\mathbf{b}_{L - k - 1}} \mathcal{L} &= \mathbf{1}^\text{T} \cdot \boldsymbol{\phi}_{L-k-1}' \cdot \prod_{l = L - k}^{L - 1}\left[\boldsymbol{\phi}_{l}' \cdot \mathbf{W}_{l}^\text{T}\right] \cdot \mathbf{e}, & \text{for } k\in\{0,1,...,L\}
    \end{align}

where the product term reduces to 1 if :math:`k = 0`.

.. note::

    .. raw:: html
    
        <details><summary>Derivation of gradients</summary>
        
    .. math::
        :nowrap:
        
        \begin{align*}
            \nabla_{\mathbf{W}_{L - 1}} \mathcal{L} &= \nabla_{\mathbf{W}_{L - 1}} \mathbf{H}_{L} \cdot \nabla_{\mathbf{H}_{L}}\mathcal{L} \\
            &= \nabla_{\mathbf{W}_{L - 1}} \mathbf{H}_{L} \cdot \mathbf{e} \\
            &= \mathbf{H}_{L - 1}^\text{T} \cdot \boldsymbol{\phi}_{L - 1}' \cdot \mathbf{e}\\
            \nabla_{\mathbf{b}_{L - 1}} \mathcal{L} &= \mathbf{1}^\text{T} \cdot \boldsymbol{\phi}_{L - 1}' \cdot \mathbf{e}\\\nonumber\\
            \nabla_{\mathbf{W}_{L - k - 1}} \mathcal{L} &= \nabla_{\mathbf{W}_{L-k-1}} \mathbf{H}_{L - k} \cdot \prod_{l = L - k}^{L - 1} \left[\nabla_{\mathbf{H}_{l}} \mathbf{H}_{l + 1}\right] \cdot \nabla_{\mathbf{H}_{L}}\mathcal{L} & \text{for } k\in\{0,1,...,L\}\\
            &= \mathbf{H}_{L - k - 1}^\text{T} \cdot \boldsymbol{\phi}_{L-k-1}' \cdot \prod_{l = L - k}^{L - 1}\left[\boldsymbol{\phi}_{l}' \cdot \mathbf{W}_{l}^\text{T}\right] \cdot \mathbf{e} & \text{for } k\in\{0,1,...,L\}\\
            \nabla_{\mathbf{b}_{L - k - 1}} \mathcal{L} &= \mathbf{1}^\text{T} \cdot \boldsymbol{\phi}_{L-k-1}' \cdot \prod_{l = L - k}^{L - 1}\left[\boldsymbol{\phi}_{l}' \cdot \mathbf{W}_{l}^\text{T}\right] \cdot \mathbf{e} & \text{for } k\in\{0,1,...,L\}
        \end{align*}

    .. raw:: html
    
        </details>
    
For simplicity we define another function alongside the sigmoid function known as the **Re**\ ctified **L**\ inear **U**\ nit (ReLU) which is defined as

    .. math::
        :nowrap:

        \begin{align}
            \phi_\text{ReLU}(z) = \max(0,z).
        \end{align}

[Hahnloser et al, 2000]\ [4]_ originally formulated the function with a biological and mathematical justification, it was then later used in object recognition by [Jarett et al, 2009]\ [5]_ and then finally popularised by both [Nair & Hinton, 2010]\ [6]_ and [Glorot & Bordes & Bengio, 2011]\ [7]_. The main benefit of the ReLU function over the sigmoid is that it suffers less from the **vanishing gradient problem**.

.. admonition:: Python
    :class: code
    
    .. raw:: html
        
        <details><summary>Base</summary>
        
    .. code-block:: python
        :linenos:
        
        from   sklearn.preprocessing   import StandardScaler
        from   sklearn.model_selection import train_test_split
        from   sklearn                 import datasets as ds
        
        from   matplotlib              import pyplot as plt
        
        import numpy as np

        # https://github.com/jordan-wei-taylor/graph
        from   graph.classes           import FCGraph

        def train_val_test_split(X, y, *args, **kwargs):
            """ Splits data into train, validation, and test sets (standardises X) """
            X_train, X_test, y_train, y_test = train_test_split(X      , y      , *args, **kwargs)
            X_train, X_val , y_train, y_val  = train_test_split(X_train, y_train, *args, **kwargs)
            scalar                           = StandardScaler().fit(X_train)
            X_train, X_val, X_test           = map(scalar.transform, [X_train, X_val, X_test])
            train, val, test                 = (X_train, y_train), (X_val, y_val), (X_test, y_test)
            return X_train, X_val, X_test, y_train, y_val, y_test, train, val, test
    
        def visualise(model, *args, **kwargs):
            """ Creates a Fully-Connected Graph """
            nfeatures    = model['layers'][0].W.shape[0]
            hidden       = [layer.W.shape[1] for layer in model['layers']]
            output       = hidden.pop(len(hidden) - 1)
            return FCGraph(nfeatures, hidden, output, *args, **kwargs)
    
        class Base():
            """
            Generic Base class
            """
            def __init__(self, params, exceptions = ['self', '__class__']):
                super().__init__()
                self._params = {k : v for k, v in params.items() if k not in exceptions}
                
            def __repr__(self):
                params = repr(self._params)[1:-1]
                return f'{self.__class__.__name__}({params})'
            
            def __getitem__(self, keys):
                return self._params[keys] if isinstance(keys, str) else [self._params[key] for key in keys]
    
    .. raw:: html
    
        </details>

    .. raw:: html
        
        <details><summary>Activation Class</summary>
        
    .. code-block:: python
        :linenos:
        
        class Activation(Base):
            """
            Collection of activations
            
            Parent class of Layer(Activation)
            """
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._activations       = dict(sigmoid = self.sigmoid, relu = self.relu, softmax = self.softmax)
                self._activations[None] = self.identity
                
            def sigmoid(self):
                def f(z): return 1 / (1 + np.exp(-z))
                def g(f, final): return f if final else f * (1 - f) # math reduces to f if final layer
                return f, g
            
            def relu(self):
                def f(z):return np.maximum(0, z)
                def g(f, **kwargs):return f > 0
                return f, g
            
            def softmax(self):
                def f(z):
                    e = np.exp(z - z.max(axis = -1, keepdims = True)) # Numerical stability
                    return e / e.sum(axis = -1, keepdims = True)
                def g(f, final):
                    if not final: raise Exception('Ugly gradient!') # should only be used as a final layer!
                    return f
                return f, g
            
            def identity(self):
                def f(z):return z
                def g(f, **kwargs):return np.ones_like(f)
                return f, g
    
    .. raw:: html
    
        </details>
        
    .. raw:: html
        
        <details><summary>Generic Layer Class</summary>
        
    .. code-block:: python
        :linenos:
        
        class Layer(Activation):
            """
            General Layer class
            
            Parent of Dense(Layer)
            """
            def __init__(self, units, activation):
                super().__init__(locals())
                self.f, self.g = self._activations[activation]()
                self._params   = dict(units = units, activation = activation)
                self._init     = True
                
            def init(self, X):
                # Initialise weights, bias, and their respective gradients
                self.W     = np.random.normal(scale = 0.5, size = (X.shape[1], self['units']))
                self.b     = np.zeros(self['units'])
                self.gW    = np.zeros_like(self.W)
                self.gb    = np.zeros_like(self.b)
                self._init = False
                
            def load_gradient(self, gradient, final = False):
                # Apply chain rule
                chain     = self.g(self._last, final = final)
                gradient *= chain
                
                # Increment gradients
                self.gW  += self._X.T @ gradient
                self.gb  += gradient.sum(axis = 0)
                
                # Back propagate the gradients to the previous layer
                return gradient @ self.W.T
                
            def update(self, alpha, momentum, l1, l2):
                """ Updates using backpropagation / chain rule """        
                # Add current gradients with penalty on the magnitude of W in the l1 or l2 spaces
                self.gW  += l1 * np.sign(self.W) + l2 * self.W
                
                # Update parameters
                self.W   -= alpha * self.gW
                self.b   -= alpha * self.gb
                
                # Decay gradients
                self.gW  *= momentum
                self.gb  *= momentum                                                                    
    
    .. raw:: html
    
        </details>
        
    .. raw:: html
        
        <details><summary>Dense Class</summary>
        
    .. code-block:: python
        :linenos:
        
        class Dense(Layer):
            """
            Dense (Fully-Connected) Layer
            
            Parameters
            ===============
                units      : int
                             Output dimension.
                             
                activation : str
                             Expects on of {"sigmoid", "softmax", "relu"}. If none provided, then no non-linear function will be applied on the projection.
            """
            def __init__(self, units, activation = None):
                super().__init__(units, activation)
            
            def __call__(self, X):
                
                # On first call initialise weights and bias
                if self._init:
                    self.init(X)
                    
                # Store input and output for later gradient computation
                self._X = X
                H       = self._last = self.f(X @ self.W + self.b)
                return H
    
    .. raw:: html
    
        </details>
        
    .. raw:: html
        
        <details><summary>Loss and Metrics</summary>
        
    .. code-block:: python
        :linenos:
        
        def Loss(loss):
            """ Obtains the loss and loss gradient functions """
            def mse():
                def f(y_true, y_pred):return np.mean(np.square(y_true - y_pred))
                def g(y_true, y_pred):return (y_pred - y_true) / len(y_true)
                return f, g
            
            def mae():
                def f(y_true, y_pred):return np.mean(np.fabs(y_true - y_pred))
                def g(y_true, y_pred):return np.sign(y_pred - y_true) / len(y_true)
                return f, g
            
            def cross_entropy():
                jitter = 1e-5 # small numerical offset to avoid log(0) !
                def f(y_true, y_pred):
                    # Binary
                    if y_true.shape[1] == 1:
                        l1 = np.log(y_pred[np.where(y_true == 1)] + jitter).sum()
                        l0 = np.log(1 - y_pred[np.where(y_true == 0)] + jitter).sum()
                        return (l1 + l0) / len(y_true)
                    # Categorical
                    else:
                        return np.log(y_pred[np.where(y_true == 1)] + jitter).sum() / len(y_true)
                def g(y_true, y_pred):return (y_pred - y_true) / len(y_true)
                return f, g
            
            losses = dict(mse = mse, mae = mae, cross_entropy = cross_entropy)
            assert loss in losses, f'Expected one of {set(losses)} not "{loss}"!'
            return losses[loss]

        class Metrics():
            """ Performance Metrics class """
            
            def rmse(self, *data):
                return np.sqrt([np.mean(np.square(self(X) - y)) for (X, y) in data])
            
            def mae(self, *data):
                return np.array([np.mean(np.fabs(self(X) - y)) for (X, y) in data])
            
            def cross_entropy(self, *data):
                jitter = 1e-5
                ret    = []
                for (X, y) in data:
                    hat = self(X)
                    if y.shape[1] == 1:
                        l1 = np.log(hat[np.where(y == 1)] + jitter).sum()
                        l0 = np.log(1 - hat[np.where(y == 0)] + jitter).sum()
                        ret.append((l1 + l0) / len(y))
                    else:
                        ret.append(np.log(hat[np.where(y == 1)] + jitter).sum() / len(y))
                return np.array(ret)
                        
            def acc(self, *data):
                return np.array([(self(X).argmax(axis = 1) == y.argmax(axis = 1)).mean() for (X, y) in data])
    
    .. raw:: html
    
        </details>
        
    .. raw:: html
        
        <details><summary>Multi-Layered Perceptron Class</summary>
        
    .. code-block:: python
        :linenos:
        
        class MultiLayeredPerceptron(Base, Metrics):
            """
            Multi-Layered Perceptron class
            
            Parameters
            ==================
                layers       : list
                               List of sequential Layer objects.
                               
                loss         : str
                               String of loss function to be minimised.
                               
                l1           : float
                               Weight penalty in l1 space.

                l2           : float
                               Weight penalty in l2 space.
                
                random_state : int
                               If provided, sets random number generation with numpy.random.seed for reproducibility.
            """
            def __init__(self, layers = [], loss = 'mse', l1 = 0., l2 = 0., random_state = None):
                super().__init__(locals())
                self.loss_f, self.loss_g = Loss(loss)()
                self.layers  = layers
                
            @staticmethod
            def batch(X, y, size):
                """ Yields partitions of X and y for batched learning """
                n = len(X)
                # If size is too big then output the original data
                if size is None or size >= n:
                    yield X, y
                    return
                r = np.random.permutation(n) # Randomised index
                m = n // size + 1
                for i in range(m):
                    yield X[r[i * size : (i + 1) * size]], y[r[i * size : (i + 1) * size]]
                    
            def fit(self, X, y, alpha = 1e-3, momentum = 0.7, epochs = 500, batch_size = None):
                
                # Set seed
                np.random.seed(self['random_state'])
                
                # Initialise loss function and wloss (regularisation) evaluations
                loss   = self.loss  = np.zeros(1 + epochs)
                wloss  = self.wloss = np.zeros((1 + epochs, 2)) # l1 and l2 regularisation
                for i in range(epochs):
                    
                    # Loop in batches to reduce memory cost
                    for Xb, yb in self.batch(X, y, batch_size):
                        hat      = self(Xb)
                        loss[i] += self.loss_f(yb, hat)
                        
                        # Load gradients to each layer and apply backpropagation
                        gradient = self.layers[-1].load_gradient(self.loss_g(yb, hat), final = True)
                        for layer in self.layers[-2::-1]:
                            gradient = layer.load_gradient(gradient)
                            
                    # Apply gradients and regularisation on the weights
                    for layer in self.layers:
                        W         = layer.W.flatten()
                        wloss[i] += self['l1'] * np.fabs(W).sum(), self['l2'] * np.square(W).sum()
                        layer.update(alpha, momentum, *self['l1', 'l2'])
                        
                # Compute final loss and wloss
                loss[-1] = self.loss_f(y, self(X))
                for layer in self.layers:
                    W          = layer.W.flatten()
                    wloss[-1] += self['l1'] * np.fabs(W).sum(), self['l2'] * np.square(W).sum()
                    
                return self
                    
            def __call__(self, H):
                for layer in self.layers:
                    H = layer(H)
                return H
    
    .. raw:: html
    
        </details>
        
Example: Load Boston
********************

.. include:: /data/load_boston.rst

Lets train an MLP with a single hidden layer extracting 10 features from the raw data and observe how regularisation affects model performance.

.. admonition:: Python
    :class: code
    
    .. code-block:: python
        :linenos:
        
        X, y = ds.load_boston(return_X_y = True)
        y.resize(len(y), 1)
        X_train, X_val, X_test, y_train, y_val, y_test, train, val, test = train_val_test_split(X, y, random_state = 2021)
        
        rmse = dict(l1 = [], l2 = [])
        L1   = np.linspace(0, 1, 100)
        L2   = np.linspace(0, 5, 100)

        for l1 in L1:
            layers = [Dense(10, activation = 'relu'), Dense(1)]
            model  = MultiLayeredPerceptron(layers, 'mse', l1 = l1, random_state = 2021).fit(X_train, y_train, batch_size = 10)
            rmse['l1'].append(model.rmse(train, val, test))

        for l2 in L2:
            layers = [Dense(10, activation = 'relu'), Dense(1)]
            model  = MultiLayeredPerceptron(layers, 'mse', l2 = l2, random_state = 2021).fit(X_train, y_train, batch_size = 10)
            rmse['l2'].append(model.rmse(train, val, test))
            
        for k, v in rmse.items():
            rmse[k] = np.array(v)
            
        fig, ax = plt.subplots(1, 2, figsize = (15, 4), sharey = True)

        for i, (k, l) in enumerate(zip(rmse, [L1, L2])):
            v = rmse[k]
            ax[i].plot(l, v)

            idx = v[:,1].argmin()
            ax[i].vlines(l[idx], v.min(), v.max())
            ax[i].set_title(f'MLP with $l_{i + 1}$', size = 18)
            ax[i].set_xlabel(f'$l_{i + 1}$', size = 15)
            ax[i].grid(ls = (0, (5, 5)))
            for j in range(3):
                ax[i].scatter(l[idx], v[idx,j], 50, zorder = 3)
                ax[i].annotate(f'{v[idx,j]:.2f}', (l[idx] + (j % 2) * 0.1 * l[-1] - 0.05 * l[-1], v[idx,j] - 0.15 + 0.3 * (j == 0)), ha = 'center', va = 'center', color = colors[j])
            ax[i].legend(['train', 'val', 'test'])
        plt.tight_layout()
        
.. figure:: /_static/how-to-ml/supervised-learning/neural-networks/load_boston.png
    :align: center
    :figwidth: 80 %
    
    RMSE against :math:`l_1` and :math:`l_2` regularisation.
    
.. admonition:: Python
    :class: code
    
    .. code-block:: python
    
        graph = visualise(model, bias = True, h_space = 15, vertical = True, annot = 40)
        graph.render()
    
.. figure:: /_static/how-to-ml/supervised-learning/neural-networks/regression-network.png
    :align: center
    :figwidth: 100 %
    
    Visualisation of how the features of load_boston is mapped through a single hidden layered MLP with with hidden layer size of 10.
    
Example: Load Iris
********************

.. include:: /data/load_iris.rst

Lets train an MLP with a single hidden layer extracting 10 features from the raw data and observe how regularisation affects model performance.

.. admonition:: Python
    :class: code
    
    .. code-block:: python
        :linenos:
        
        X, y = ds.load_iris(return_X_y = True)
        y    = np.eye(3)[y]
        X_train, X_val, X_test, y_train, y_val, y_test, train, val, test = train_val_test_split(X, y, random_state = 2021)

        L = np.linspace(0, 0.01, 100)

        entropy = dict(l1 = [], l2 = [])
        acc     = dict(l1 = [], l2 = [])

        for l1 in L:
            layers = [Dense(10, activation = 'relu'), Dense(3, activation = 'softmax')]
            model  = MultiLayeredPerceptron(layers, 'cross_entropy', l1 = l1, random_state = 2021).fit(X_train, y_train, alpha = 1e-1, momentum = 0.4)
            entropy['l1'].append(model.cross_entropy(train, val, test))
            acc['l1'].append(model.acc(train, val, test))
            
        for l2 in L:
            layers = [Dense(10, activation = 'relu'), Dense(3, activation = 'softmax')]
            model  = MultiLayeredPerceptron(layers, 'cross_entropy', l2 = l2, random_state = 2021).fit(X_train, y_train, alpha = 1e-1, momentum = 0.4)
            entropy['l2'].append(model.cross_entropy(train, val, test))
            acc['l2'].append(model.acc(train, val, test))
            
        for k in acc:
            entropy[k] = np.array(entropy[k])
            acc[k]     = np.array(acc[k])
            
        fig, ax = plt.subplots(2, 2, figsize = (12, 8), sharey = 'row', sharex = True)
        ax      = ax.flatten()

        for i in range(2):
            e   = entropy[f'l{i + 1}']
            a   = acc[f'l{i + 1}']
            idx = e[:,1].argmax()
            for j, d in enumerate([e, a]):
                ax[i + 2 * j].plot(L, d)
                ax[i + 2 * j].vlines(L[idx], d.min(), d.max(), ls = '--')
                for k in range(3):
                    ax[i + 2 * j].scatter(L[idx], d[idx,k], 50, colors[k], zorder = 3)
                    ax[i + 2 * j].annotate(f'{d[idx, k]:.2f}',
                                           (L[idx] - 0.001 + 0.0015 * i + 0.0005 * (j * i) + 0.0005 * (j * (i == 0) * (k == 2)),
                                            d[idx,k] - 0.005 + 0.01 * (j * (1 - i)) * (k == 2)),
                                           color = colors[k], ha = 'center', va = 'center')
                ax[i + 2 * j].grid(ls = (0, (5, 5)))
            ax[i    ].set_title(f'$l_{i + 1}$-Regularisation')
        ax[0].set_ylabel('Cross Entropy')
        ax[2].set_ylabel('Accuracy')
        for i in range(2):
            ax[i + 2].set_xlabel(f'$l_{i + 1}$')
            
.. figure:: /_static/how-to-ml/supervised-learning/neural-networks/load_iris.png
    :align: center
    :figwidth: 80 %
    
    Cross Entropy and Accuracy against :math:`l_1` and :math:`l_2` regularisation.
    
.. admonition:: Python
    :class: code
    
    .. code-block:: python
    
        graph = visualise(model, bias = True, h_space = 8, vertical = True, annot = 30)
        graph.render()
    
.. figure:: /_static/how-to-ml/supervised-learning/neural-networks/classification-network.png
    :align: center
    :figwidth: 100 %
    
    Visualisation of how the features of load_iris is mapped through a single hidden layered MLP with with hidden layer size of 10.

Bibliography
************

.. [1] W. S. McCulloch & W. Pitts, *A Logical Calculus of the Ideas Immanent in Nervous Activity*, 1943
.. [2] F. Rosenblatt, **Principles of Neurodynamics**: *Perceptrons and the Theory of Brain Mechanisms*, 1961
.. [3] D. E. Rumelhart & G. E. Hinton & R. J. Williams, **Nature**, *Learning Representations by Back-Propagating Errors*, 1986
.. [4] R. H. R. Hahnloser et al, **Nature**, *Digitial Selection and Analogue Amplification Coexist in a Cortex-Inspired Silicon Circuit*, 2000
.. [5] K. Jarret et al, **IEEE**, *What is the Best Multi-Stage Architecture for Object Recognition?*, 2009
.. [6] V. Nair & G. E. Hinton, **ICML**, *Rectified Linear Units Improve Restricted Boltzmann Machines*, 2010
.. [7] X. Glorot & A. Bordes & Y. Bengio, **Journal of Machine Learning Research**, *Deep Sparse Rectifier Neural Networks*, 2011






