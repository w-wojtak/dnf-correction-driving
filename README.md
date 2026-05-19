# A brain-inspired approach to learning and recalling temporal sequences using Dynamic Neural Fields (DNFs).

## What are Dynamic Neural Fields?
Dynamic Neural Fields are continuous neural networks that represent information as localized activation patterns (peaks or "bumps") across a feature space. Think of them as a continuous version of traditional neural networks where:

* **Peak position** encodes _what_ (which item/destination/action)
* **Peak amplitude** encodes _when_ (temporal order in sequence)

## How Learning Works
1. Observation Phase (`learning.py`)
When observing a sequence (e.g., a driver's Monday routine: coffee → work → gym → home):
```
Time 0: Input at coffee location (x=-60)
Time 1: Input at work location (x=-30)
Time 2: Input at gym location (x=0)
Time 3: Input at home location (x=30)
```

