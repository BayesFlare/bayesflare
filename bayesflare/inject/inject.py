def inject_model(lightcurve, model, instance=0):

    """
    Function to inject a model instance into a Light curve

    Parameters
    ----------
    lightcurve : BayesFlare Lightcurve instance
       The light curve which the model should be injected into.

    model : BayesFlare Model instance
       The model which should be injected into `lightcurve`.

    instance : int
       The component of `model` which should be injected.

    Returns
    -------
    lightcurve : BayesFlare Lightcurve instance
       The light curve with an injected model.

    """

    M = model(instance).clc
    L = lightcurve.clc

    lightcurve.clc = L + M
    lightcurve.original = M
    return lightcurve


