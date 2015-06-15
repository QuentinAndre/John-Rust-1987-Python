.. highlight:: sh
==============
 Introduction
==============

:Date: April 11, 2015
:Version: 0.9.0
:Authors: Quentin ANDRE, quentin.andre@insead.edu
:Web site: https://github.com/QuentinAndre/John-Rust-1987-Python
:Copyright: This document has been placed in the public domain.
:License: John-Rust-1987-Python is released under the MIT license.

Purpose
=======

John-Rust-1987-Python provides a Python implementation of the single agent dynamic choice model of bus engine replacement described by John Rust in his 1987 article *"Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher"*.

Content
=======

1. *Data Generation and Likelihood Fit.ipynb**: An IPython notebook describing the different steps used to:
 * Generate the bus mileage data
 * Compute the forward-looking (dynamic) utility of the agent
 * Generate four datasets of bus engine replacement:
  * One assuming a linear maintainance cost (Lin_Dataset.csv)
  * One assuming a quadratic maintainance cost (Quad_Dataset.csv)
  * One assuming an exponential maintainance cost (Exp_Dataset.csv)
  * One assuming a logarithmic maintainance cost (Log_Dataset.csv)
 * Estimate the parameters of this bus replacement pattern
 * Plot the results of this estimation against the true underlying parameters.

2. **dynamiclogit.py**: A library containing a DynamicUtility class, which can be used as a statistical workbench to estimate single-agent dynamic choice models. The state transition probabilities of the buses and the form of the maintenance cost are supplied at the class initialization as an external parameters, which allows to test a wide range of specifications.

Installation
============

Dependencies
------------
This code has been tested in Python 3.4, using the Anaconda distribution:
 * `The Anaconda distribution for Python 3.4 <http://continuum.io/downloads#py34>`_

Download
--------

* Using git:
 * git clone https://github.com/QuentinAndre/John-Rust-1987-Python.git

* Download the master branch as a zip: 
 * https://github.com/QuentinAndre/John-Rust-1987-Python/archive/master.zip


References
==========
Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. Econometrica: Journal of the Econometric Society, 999-1033.
