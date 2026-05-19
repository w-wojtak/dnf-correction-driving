# A brain-inspired approach to learning and recalling temporal sequences using Dynamic Neural Fields (DNFs).

## What are Dynamic Neural Fields?
Dynamic Neural Fields are continuous neural networks that represent information as localized activation patterns (peaks or "bumps") across a feature space. Think of them as a continuous version of traditional neural networks where:

* **Peak position** encodes _what_ (which item/destination/action)
* **Peak amplitude** encodes _when_ (temporal order in sequence)

## How Learning Works
### 1. Observation Phase (`src/sequence_learning.py`)
When observing a sequence (e.g., a driver's Monday routine: coffee → work → gym → home):
```
Time 0: Input at coffee location (x=-60)
Time 1: Input at work location (x=-30)
Time 2: Input at gym location (x=0)
Time 3: Input at home location (x=30)
```

**Key mechanism: Threshold Accommodation**

* Each input creates a peak in the destination memory field
* A threshold variable ``h increases continuously over time
* Earlier inputs accumulate more `h` → higher final peak
* Result: activation gradient where peak height = temporal order

```
Final memory state:
  coffee peak: amplitude 4.5 (visited first  → highest)
  work peak:   amplitude 3.8 (visited second)
  gym peak:    amplitude 2.9 (visited third)
  home peak:   amplitude 2.0 (visited last   → lowest)
  ```

**The memory field stores both:**

* Spatial: peaks at x=-60, -30, 0, 30 (which destinations)
* Temporal: amplitude gradient (their order)

### 2. Duration Encoding
A separate field encodes the total routine duration (e.g., 24 hours for a daily routine). This acts as a "timer" that adjust recall onset.



## How Recall Works (`src/sequence_recall.py`)

Recall is an autonomous competitive process:

The Dynamics
* Initialization: Load learned memory into prediction field u_dest_pred
* Ramp mechanism: Threshold accommodation (h_u_dest) increases uniformly
* Winner-take-all: Highest peak (earliest destination) crosses threshold first
* Inhibition: Working memory field u_wm stores activated peak and inhibits that location
* Repeat: Next-highest peak now wins, process continues

```
t=1.8:  coffee peak crosses threshold  → "predict coffee arrival"
t=4.2:  work peak crosses threshold    → "predict work arrival"  
t=7.5:  gym peak crosses threshold     → "predict gym arrival"
t=10.8: home peak crosses threshold    → "predict home arrival"
```

The time between threshold crossings reflects the amplitude differences, which encode the original temporal structure.

