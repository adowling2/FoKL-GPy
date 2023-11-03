# FoKL
Karhunen Loève decomposed Gaussian processes with forward variable
selection. Use this package for scalable GP regression and fast
inference on static and dynamic datasets.

## Setup
To install, use 'pip install FoKL' or clone this repo. Once installed, import into your environment with:
```
from FoKL import FoKLRoutines
```
If integrating, then include:
```
from FoKL import GP_Integrate
```
Now you are ready to begin creating your model, which can be initialized with:
```
model = FoKLRoutines.FoKL()
```
If intending to override the default hyperparameters, then you can include keywords in the model's initialization. For example:
```
model = FoKLRoutines.FoKL(btau=1000, draws=2000, way3=1)
```
Alternatively, hyperparameters can be redefined or updated with:
```
model.btau  = 1000
model.draws = 2000
model.way3  = 1
```
The above is useful for performing sweeps through hyperparameters without needing to initialize a new model (i.e., a new Python class) for each new combination of hyperparameters.

The default hyperparameters and their keywords are as follows:
```
phis       = getKernels.sp500()
relats_in  = []
a          = 4
b          = f(a, data)
atau       = 4
btau       = f(atau, data)
tolerance  = 3
draws      = 1000
gimmie     = False
way3       = False
threshav   = 0.05
threshstda = 0.5
threshstdb = 2
aic        = False
```
A description of each hyperparameter is listed in the function documentation.

## Training
Call the 'fit' function to train the FoKL model on all of 'data':
```
betas, mtx, evs = model.fit(inputs, data)
```
Optionally, include the keyword 'train' as the percentage of 'data' to use for training (e.g., 80%):
```
betas, mtx, evs = model.fit(inputs, data, train=0.8)
```
The console will display the index and bic of the model being built in real time. Once completed, the model can be validated with the 'coverage3' function:
```
meens, bounds, rmse = model.coverage3()
```
By default, 'coverage3' predicts output values for 'model.inputs', which is just the normalized and properly formatted 'inputs' provided in 'fit'. If validating visually, then a sorted plot of the test set (for 'train' < 1) tends to be most insightful:
```
model.coverage3(inputs=model.testinputs, data=model.testdata, plot='sorted', bounds=1, legend=1)
```
Note 'data' must correspond to the set used for 'inputs' to calculate the model's RMSE, which is the third positional output of 'coverage3'.

If not requiring the RMSE (or a plot), then the 'evaluate' function can be used to bypass 'coverage3' so that any inputs can be evaluated. In other words, 'coverage3' takes 'evaluate' one step farther by returning the RMSE but is limited to validation testing only since the corresponding data must also be provided with the inputs. To evaluate the inputs 'userinputs' for which the output data is not known, use the following to predict the data ('meen') and confidence bounds ('bounds'):
```
meen, bounds = model.evaluate(userinputs)
```
Note 'userinputs' will be automatically normalized to the same scale as the training inputs that the model was fitted to, as well as automatically formatted. In the rare case that 'userinputs' is already normalized and formatted properly, then this automatic treatment of 'userinputs' can be turned off with the following keyword:
```
meen, bounds = model.evaluate(userinputs, nform=0)
```
As an appended side note, the following attributes were added to your FoKL class 'model' after calling 'fit' which may be useful during user post-processing:
```
model.inputs         == all normalized inputs w/o outliers (i.e., model.traininputs plus model.testinputs)
model.data           == all data w/o outliers (i.e., model.traindata plus model.testdata)

model.betas          == betas
model.mtx            == mtx
model.evs            == evs

model.rawinputs      == all normalized inputs w/ outliers == user's 'inputs' but normalized and formatted
model.rawdata        == all data w/ outliers              == user's 'data' but formatted
model.traininputs    == train set of model.inputs
model.traindata      == train set of model.data
model.testinputs     == test set of model.inputs
model.testdata       == test set of model.data
model.normalize      == [min, max] factors used to normalize user's 'inputs' to 0-1 scale of model.rawinputs
model.outliers       == indices removed from model.rawinputs and model.rawdata as outliers
model.trainlog       == indices of model.inputs used for model.traininputs
model.testlog        == indices of model.data used for model.traindata

model.inputs_np      == model.inputs as a numpy array of timestamps x input variables
model.rawinputs_np   == model.rawinputs as a numpy array of timestamps x input variables
model.traininputs_np == model.traininputs as a numpy array of timestamps x input variables
model.testinputs_np  == model.testinputs as a numpy array of timestamps x input variables
```
To remove all of the above attributes so that only the hyperparameters remain, most importantly so that 'betas' does not influence the training of a new model, use:
```
model.clear()
```

## Differentiation
FoKL can be used to calculate the derivatives of the states (i.e., input variables). By default, 'model.inputs_np' is used with a finite differences approximation of second order. The default functionality is as follows:
```
dState = bss_derivatives()
```
In the following example, the first state's derivative is evaluated using a second order approximation whereas the second state uses a first order approximation. The keywords are as follows: 
```
dStates = bss_derivatives(inputs=[State1,State2], derv=[2,1])
```
Note the states provided, if not default, should follow a dynamic system such that the datapoints are sampled at regular intervals through time. Setting 'inputs=model.testinputs', for example, may produce erroneous results due to the selection of the test set not accounting for regular intervals of time.

## Integration
As discussed in the previous section, FoKL can be used to model state derivatives and thus contains an integration method of these states using an RK4. Due to each state being modeled independently, the same functionality cannot be used. For the case of two states, 'State1' and 'State2', with the same inputs:
```
model = FoKLRoutines.FoKL()

dStates = [dState1, dState2]
betas = []
mtx = []
for i in range(2):
    betas_i, mtx_i, _ = model.fit(inputs, dStates[i])
    betas.append(betas_i)
    mtx.append(mtx_i)
    model.clear()
```
After fitting the above state derivatives, call the 'GP_Integrate' function to integrate:
```
T, Y = GP_Integrate([np.mean(betas[0],axis=0),np.mean(betas[1],axis=0)], [mtx[0],mtx[1]], utest, norms, phis, start, stop, ic, stepsize, used_inputs)
```
Alternatively, multiple separate FoKL classes can be created to achieve the same result:
```
model1 = FoKLRoutines.FoKL()
model2 = FoKLRoutines.FoKL()

betas1, mtx1, _ = model1.fit(inputs, dState1)
betas2, mtx2, _ = model2.fit(inputs, dState2)

T, Y = GP_Integrate([np.mean(betas1,axis=0),np.mean(betas2,axis=0)], [mtx1,mtx2], utest, norms, phis, start, stop, ic, stepsize, used_inputs)
```
See 'GP_intergrate_example.py' for an example.

## Development

More sophisticated outlier removal methods are currently in development, but for demonstration purposes the following will search through 'data' and remove any points with a z-score greater than 4:
```
model.fit(inputs, data, CatchOutliers='Data', OutliersMethod='Z-Score', OutliersMethodParams=4)
```
Also in development are additional methods for splitting 'data' into test/train sets, beyond the current method which is limited to a random split.

It is also intended for the 'evaluate' function to be capable of providing derivatives, sampling through 'betas' of the model, and evaluating at user-defined 'betas'.

## Citations
Please cite: K. Hayes, M.W. Fouts, A. Baheri and
D.S. Mebane, "Forward variable selection enables fast and accurate
dynamic system identification with Karhunen-Loève decomposed Gaussian
processes", arXiv:2205.13676

Credits: David Mebane (ideas and original code), Kyle Hayes
(integrator), Derek Slack (Python porting), Jacob Krell (Python v3 dev.)

Funding provided by National Science Foundation, Award No. 2119688

