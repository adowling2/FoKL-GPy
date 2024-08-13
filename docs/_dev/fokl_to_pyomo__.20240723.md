```python
fokl_to_pyomo(self, xvars, yvar, m=None, t=None, scenarios=None)
```

---

```python
s = scenarios

if t is None and s is None:
    ts = None
else:  
    if t is None:
        ts = s
    elif s is None:
        ts = t
    else:
        ts = [t, s]
```

---

If   ```xvars``` or ```yvar``` are strings,

```python
for j in len(xvars):
    if isinstance(xvars[j], str):
        xvars[j] = pyo.Var(ts, bounds=self.minmax[j])

if isinstance(yvar, str):
    yvar = pyo.Var(ts)
```

---

If ```s```,

- ```m.GPi[(t), s]   # == pyo.Constraint(yvar == m.GPi_y)```
- ```m.GPi_y[(t), s]```
- ```m.GPi_beta[s, term]```
    - assume the most recent GP draws are the Pyomo scenarios; that is, ```= self.betas[-(s + 1), term]```.

Else,

- ```m.GPi[(t)]   # == pyo.Constraint(yvar == m.GPi_y)```
- ```m.GPi_y[(t)]```
- ```m.GPi_beta[term]```
    - assume the average of the GP draws are the single Pyomo scenario; that is, ```= np.mean(self.betas, axis=0)[term]```.

Note ```xvars``` and ```yvar``` are not required to be indexed by ```s```.

---

If ```t```,

- ```m.GPi[t, (s)]  # == pyo.Constraint(yvar == m.GPi_y)```
- ```m.GPi_y[t, (s)]```

Note at least one variable in ```[xvars, yvar]``` must be indexed by ```t```.



