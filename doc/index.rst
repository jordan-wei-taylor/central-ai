##############
Central AI
##############

Welcome to Central AI, an educational website to sharpen your machine learning knowledge!

Background
***********

Hi, my name is Jordan and I am the author of Central AI, a resource that aims to centralise knowledge on Data Science - specifically Machine Learning and Reinfocement Learning. After completing my undergraduate degree in Actuarial Sciences at the University of East Anglia in 2015, I worked under the asset management department within NHS Property Services as an analyst until late 2017. I then completed my MSc Data Science at the University of Bath and decided to pursue a PhD. Since then I have completed an MRes in Statistical Applied Mathematics and am currently on the Statistical Applied Mathematics Bath (SAMBa) CDT. I have found during my time as a Data Science student, and now tutoring it, that easy to access online material for a lot of machine learning concepts are not well covered nor centralised. This has motivated me to attempt to fill in the gaps with tutorials, where I implement these concepts and algorithms in Python, all in one central location.

I have a few tutorial channels, each with a different theme or purpose.


How to: Machine Learning
-------------------------

A beginner's guide to machine learning designed for those that know a little Python and some key terms. Suited for those in education who want to understand the algorithms. A **from scratch** attitude is adopted here whereby most things will be built using **numpy** and **scipy** instead of importing off-the-shelf algorithms from **sklearn** (scikit-learn).

How to: Reinforcement Learning
-------------------------------

An introduction to Reinforcement Learning starting from what it is, and going from the basics with Markov Decision Processes to function approximation with Neural Networks. Tabular methods are implemented in **numpy** whilst function approximation is in **PyTorch**.

Advanced Applications
---------------------

Building further on the *How to* tutorials series, apply ML and RL techniques on more interesting and more realistic problems. Most implementations will be in **PyTorch**.

Theories
-----------------

A bit more math-heavy, this set of tutorials looks to answer the theoretical problems present in machine learning.

.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: How to Tutorials
    :titlesonly:

    how-to-ml/index

.. toctree::
    :hidden:
    :maxdepth: 2

    how-to-rl/index

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Advanced Applications

    advanced-applications/ml/index
    advanced-applications/rl/index

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Theories
    :titlesonly:

    theories/regression
    theories/classification

.. toctree::
    :hidden:
    :glob:
    :maxdepth: 1
    :caption: Appendix
    :titlesonly:

    appendix/*
