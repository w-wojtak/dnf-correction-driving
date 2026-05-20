# Learning and recalling temporal sequences using Dynamic Neural Fields (DNFs)

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
* A threshold variable `h` increases continuously over time
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
* Initialization: Load learned memory into prediction field `u_dest_pred`
* Ramp mechanism: Threshold accommodation (`h_u_dest`) increases uniformly
* Winner-take-all: Highest peak (earliest destination) crosses threshold first
* Inhibition: Working memory field `u_wm` stores activated peak and inhibits that location
* Repeat: Next-highest peak now wins, process continues

```
t=1.8:  coffee peak crosses threshold  → "predict coffee arrival"
t=4.2:  work peak crosses threshold    → "predict work arrival"  
t=7.5:  gym peak crosses threshold     → "predict gym arrival"
t=10.8: home peak crosses threshold    → "predict home arrival"
```

The time between threshold crossings reflects the amplitude differences, which encode the original temporal structure.


## Sequence Correction via Verbal Feedback (`src/sequence_tuning.py`)

After learning a routine, the system can be **rapidly adapted** through natural language corrections without re-demonstration or waiting for new data accumulation.

### Correction Mechanism

Each correction type is implemented as a dedicated **feedback field** that evolves via DNF dynamics. When triggered (e.g., driver says *"skip gym"*), the field:

1. Receives localized Gaussian input at the target destination's position
2. Evolves autonomously following standard DNF dynamics
3. Produces thresholded output defining the spatial region to modify
4. Drives **structured, local modification** of the destination memory field

This implements corrections through **field-mediated memory updates** rather than symbolic rules.

### Correction Types

| Command | Example | Effect on Memory | Result in Recall |
|---------|---------|------------------|------------------|
| **SKIP** | *"skip gym"* | Suppress peak → flatten activation | Destination not predicted |
| **EARLY** | *"arrive at work earlier"* | Weaken peak → decrease amplitude | Recalls earlier in sequence |
| **LATE** | *"leave gym later"* | Strengthen peak → increase amplitude | Recalls later in sequence |
| **SWAP** | *"gym before work now"* | Exchange peak amplitudes | Reverses temporal order |
| **LOCK** | *"always predict home last"* | Protect region from changes | Immune to future corrections |

### Key Principles

**Why this works:**

- **Peak amplitude = temporal order**: Weakening a peak makes it fire earlier during recall's ramp-to-threshold dynamics
- **Spatial locality**: Corrections affect only the targeted destination(s)
- **Immediate effect**: No retraining—memory is directly modified, active next recall
- **Composable**: Multiple corrections can be applied sequentially (e.g., LOCK home → SKIP gym → EARLY work)



### Running Corrections
```
# In `src/sequence_tuning.py`, set desired feedback:

# Skip a destination
human_feedback = (FeedbackType.SKIP, "gym")

# Adjust timing
human_feedback = (FeedbackType.EARLY, "work")

# Reorder sequence
human_feedback = (FeedbackType.SWAP, "work", "gym")

# Protect critical destination
human_feedback = (FeedbackType.LOCK, "home")
```



### Correction Examples

 
---

#### SKIP: Remove a Destination

<p align="center">
  <img src="docs/images/correction_skip.png" width="90%">
</p>

**Command:** *"skip gym"*

**Mechanism:** SKIP field suppresses the gym peak to baseline level.

**Result:** System no longer predicts gym visits. Routine: `coffee → work → gym → home` becomes `coffee → work → home`

---

#### EARLY: Predict Earlier Arrival

<p align="center">
  <img src="docs/images/correction_early.png" width="90%">
</p>

**Command:** *"arrive at work earlier"*

**Mechanism:** EARLY field weakens work peak amplitude (3.85 → 3.35).

**Result:** Lower amplitude → crosses threshold earlier in recall → predicts earlier arrival time.

---

#### LATE: Predict Later Arrival

<p align="center">
  <img src="docs/images/correction_late.png" width="90%">
</p>

**Command:** *"leave gym later"*

**Mechanism:** LATE field strengthens gym peak amplitude (2.90 → 3.40).

**Result:** Higher amplitude → crosses threshold later in recall → predicts later departure.

---

#### SWAP: Reverse Order

<p align="center">
  <img src="docs/images/correction_swap.png" width="90%">
</p>

**Command:** *"gym before work now"*

**Mechanism:** SWAP field exchanges work ↔ gym peak amplitudes.

**Result:** Routine: `coffee → work → gym → home` becomes `coffee → gym → work → home`

---

### LOCK: Protect Destination

<p align="center">
  <img src="docs/images/correction_lock.png" width="90%">
</p>

**Command:** *"always predict home last"*

**Mechanism:** LOCK field creates a mask preventing future modifications at home location.

**Result:** Home destination immune to SKIP/EARLY/LATE corrections (safety-critical preservation).

---

### Summary Table

| Correction | Peak Amplitude Change | Recall Effect |
|------------|----------------------|---------------|
| **SKIP**   | → 0 (baseline)       | Not predicted |
| **EARLY**  | ↓ (weaken)           | Fires earlier |
| **LATE**   | ↑ (strengthen)       | Fires later   |
| **SWAP**   | ↔ (exchange)         | Order reversed|
| **LOCK**   | — (protected)        | Immutable     |