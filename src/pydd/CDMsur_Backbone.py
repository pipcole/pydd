import numpy as np
               
from utils_surr import *
import os
import pySurrogate

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import rompy as rp

# A new gegularized grid.
N_surrogate = 1000
print(N_surrogate)
x0 = np.logspace(-5, 0, N_surrogate)
croppedDistance = 1.2 # Crop all signal to 1.1 *r5 approx 10yrs

# path = "../Cold Dark Matter Waveforms/HaloFeedbackRuns_Dec2021/"
path = "NewData/"

def transformInput(input: np.array) -> np.array:
  """ Transforms the input vector to the desired shape or units for the model. """
  m1, m2, rho6, gammasp = input.T

  # Train with the logarithm of most of the values to allow flexibility with smaller numbers.
  X = np.vstack([np.log10(m1), np.log10(m2), np.log10(rho6), gammasp]).T

  return X

def trainSurrogate():
    data0 = []
    input = []

    all_files = os.listdir(path)

    for filepath in all_files:
        if filepath[0] == ".": continue
        _loaded = np.loadtxt(path+ filepath)
        
        params = filepath.split("_")
        m1 = float(params[2]); m2 = float(params[4]); rho6 = float(params[6]); gamma = float(params[8][:-7])

        # Discard a few weird samples
        r5 = getVacuumMergerDistance(m1, m2, 5 *yr)
        t, r, _, rheff = _loaded[_loaded.T[1] <= croppedDistance *r5].T
        if np.any(np.gradient(rheff, r) > 0): continue
        
        data0.append(np.loadtxt(path+ filepath))
        input.append([m1, m2, rho6, gamma]) # M_sun, M_sun, 1e16 M_sun/pc3, _

    input = np.vstack(input)
    m1, m2, rho6, gammasp = input.T

    # Place in new grid and rescale stuff
    r5 = getVacuumMergerDistance(m1, m2, 5 *yr)
    
    # Crop data before the at croppedDistance *r5 = 25 years before from merger.
    data = []
    for i, point in enumerate(data0):
        t, r, _, rheff = point[point.T[1] <= croppedDistance *r5[i]].T # Keep data below croppedDistance r5yr to avoid fast depletion initialization.
        
        # Redefine the separation as a normalized x:
        a = r[-1]; b = croppedDistance *r5[i]
        x = (r -a)/(b -a)
        
        # Map to a new grid of size N.
        y_ = interp1d(x, rheff, bounds_error = False, fill_value = "extrapolate", kind = "linear")(x0)
        
        data.append(y_)

    data = np.vstack(data)
    
    # Construct the dephasing
    risco = getRisco(m1)
    r = croppedDistance *getVacuumMergerDistance(m1, m2, 5 *yr) -risco
    r = x0 *r.reshape(-1, 1) +np.ones((len(r), len(x0))) *risco.reshape(-1, 1)

    dPhase = np.array([ np.flip(getDephasingFromDensity(r[i], data[i], m1[i], m2[i])[2]) for i in range(len(m1))])
    
    # decoder =  pySurrogate.Decoder(np.log10(dPhase[:, 1:].astype(float)))
    # decoder.physicalTransform = lambda x, input: np.power(10, x)
    integration = rp.Integration([x0.min(), x0.max()], num = len(x0) -1, rule = "trapezoidal")

    decoder = rp.Surrogate(integration)
    decoder.MakeROM(np.log10(dPhase[:, 1:].astype(float)), 0, 1e-8)
    print(f"Base constructed with {decoder.ei.data.shape[0]} waveforms")
    
    decoder =  pySurrogate.Decoder(projection_coeffs = decoder.ei.data.T, basis = decoder.ei.B)
    decoder.physicalTransform = lambda x, input: np.power(10, x) # Undo any other transformation.

    # Define the scalers
    scalerX, scalerY = StandardScaler(), StandardScaler()

    # Initialize the pySurrogate.Initializer object.
    sur_init = pySurrogate.Initializer(transformInput, scalerX, scalerY)
    sur_decoder = decoder
    
    # Construct the input and output vectors of the model.
    X = sur_init.inputTranformer(input)
    Y = sur_decoder.projection_coeffs

    X, Y = sur_init.fitScalerXY(X, Y)
    
    kernel = RBF() *C() +WhiteKernel()

    GP = GaussianProcessRegressor(kernel, alpha = 1e-10, n_restarts_optimizer = 10)
    GP.fit(X, Y)
    print(GP.score(X, Y))
    
    sur_regr = pySurrogate.Regressor(GP, GP.predict)
    
    model = pySurrogate.Surrogate(sur_init, sur_decoder, sur_regr)
    
    return model, input

def getPrediction(model, input):
    """ Returns frequency, full phase (dephasing with vauum) time-frequency evolution and \ddot(Phase)."""
    dPhase = np.insert(model(input)[0], 0, 0) # Append a last zero at the end.
    m1, m2, _, _ = input

    # Construct the dephasing
    risco = getRisco(m1)
    r = croppedDistance *getVacuumMergerDistance(m1, m2, 5 *yr) -risco
    r = x0 *r +risco

    fGW = 2/getPeriodFromDistance(r, (m1 +m2))
    
    # Vacuum phase
    Phase_V = getVacuumPhase(fGW, m1, m2)
    Phase = Phase_V -dPhase
    
    # Construct other waveform-building quantities as well.
    dPhase_df = np.gradient(Phase, fGW)
    t_f = cumtrapz(dPhase_df/fGW, fGW, initial = 0) / (2 *np.pi)
    ddPhase = -4 *np.pi**2 *fGW /dPhase_df
    
    return fGW, Phase, t_f, ddPhase

def getDephasing(model, input):
    dPhase = np.insert(model(input)[0], 0, 0) # Append a last zero at the end.
    m1, m2, rho6, gammasp = input

    # Construct the dephasing
    risco = getRisco(m1)
    r = croppedDistance *getVacuumMergerDistance(m1, m2, 5 *yr) -risco
    r = x0 *r +risco

    fGW = 2/getPeriodFromDistance(r, (m1 +m2))
    
    return fGW, dPhase