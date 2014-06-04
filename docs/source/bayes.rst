BayesFlare Bayesian Functions
##########################

The Bayes and Parameter estimation classes
==========================================

.. automodule:: bayesflare.stats.bayes
   :members:

.. automodule:: pyflare.stats.bayes
   :members:

General functions
=================

.. c:macro:: LOG2PI

   The natural logarithm of :math:`2\pi` defined using the GSL macros `(M_LN2 + M_LNPI)`

.. c:macro:: LOGPI_2

   The natural logarithm of :math:`\pi/2` defined using the GSL macros `(M_LNPI - M_LN2)`

.. c:function:: double log_marg_amp_full_C(int Nmodels, double modelModel[], double dataModel[], double sigma, unsigned int lastHalfRange)

   This function calculates the log-likelihood ratio for a signal model (consisting of a number
   `Nmodels` components, each with an amplitude that is marginalised over) compared to a pure
   Gaussian noise model. For all bar the last model component the amplitudes are analytically
   marginalised between -infinity and +infinity, whilst if `lastHalfRange` is true the final
   amplitude will be marginalised between 0 and infinity (i.e. it must be positive).

   For a more complete description of this function see [1]_ or Algorithm 1 in [2]_.

   Parameters
   ----------
   Nmodels : int
       The number of model components.
   modelModel : double array
       A flattened 1D array consisting of the summed cross terms for each model component [size:
       `Nmodels` * `Nmodels`].
   dataModel : double array
       A 1D array of the summed data crossed with each model component [size: `Nmodels`]
   sigma : double
       The underlying Gaussian noise standard deviation
   lastHalfRange : bool
       A boolean saying whether the final model component amplitudes should be marginalised over
       the full -infinity to infinity range (False), or between 0 and inf (True).

   Returns
   -------
   logL : double
       The log-likelihood ratio marginalised over the model component amplitudes.

   References
   ----------

    The natural logarithm of :math:`2\pi` defined using the GSL macros `(M_LN2 + M_LNPI)`

.. c:macro:: LOGPI_2
    The natural logarithm of :math:`\pi/2` defined using the GSL macros `(M_LNPI - M_LN2)`

.. c:function:: double log_marg_amp_full_C(int Nmodels, double modelModel[], double dataModel[],
                double sigma, unsigned int lastHalfRange)
    This function calculates the log-likelihood ratio for a signal model (consisting of a number
    `Nmodels` components, each with an amplitude that is marginalised over) compared to a pure
    Gaussian noise model. For all bar the last model component the amplitudes are analytically
    marginalised between -infinity and +infinity, whilst if `lastHalfRange` is true the final
    amplitude will be marginalised between 0 and infinity (i.e. it must be positive).

    For a more complete description of this function see [1]_ or Algorithm 1 in [2]_.

    Parameters
    ----------
    Nmodels : int
        The number of model components.
    modelModel : double array
        A flattened 1D array consisting of the summed cross terms for each model component [size:
        `Nmodels` * `Nmodels`].
    dataModel : double array
        A 1D array of the summed data crossed with each model component [size: `Nmodels`]
    sigma : double
        The underlying Gaussian noise standard deviation
    lastHalfRange : bool
        A boolean saying whether the final model component amplitudes should be marginalised over
        the full -infinity to infinity range (False), or between 0 and inf (True).

    Returns
    -------
    logL : double
        The log-likelihood ratio marginalised over the model component amplitudes.

    References
    ----------

    .. [1] http://mattpitkin.github.io/amplitude-marginaliser
    .. [2] Pitkin, Williams, Fletcher and Grant, 2014, `arXiv:14XX.XXXX
<http://arxiv.org/abs/14XX.XXX>`.

.. c:function:: double log_marg_amp_except_final_C(int Nmodels, double modelModel[],
                                                   double dataModel[], double sigma)

   This function calculates the log-likelihood ratio for a signal model (consisting of a number
   `Nmodels` components) compared to a pure Gaussian noise model. For all bar the last model
   component the amplitudes are analytically marginalised between -infinity and +infinity.

   For a more complete description of this function see [1]_ or Algorithm 2 in [2]_.

   Parameters
   ----------
   Nmodels : int
       The number of model components.
   modelModel : double array
       A flattened 1D array consisting of the summed cross terms for each model component [size:
       `Nmodels` * `Nmodels`].
   dataModel : double array
       A 1D array of the summed data crossed with each model component [size: `Nmodels`]
   sigma : double
       The underlying Gaussian noise standard deviation

   Returns
   -------
   logL : double
       The log-likelihood ratio marginalised over the model component amplitudes.

   References
   ----------
   .. [1] http://mattpitkin.github.io/amplitude-marginaliser
   .. [2] Pitkin, Williams, Fletcher and Grant, 2014, `arXiv:14XX.XXXX
<http://arxiv.org/abs/14XX.XXX>`.

.. automodule:: bayesflare.stats.general
    :members:

