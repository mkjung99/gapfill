# GapFill
GapFill: A Python module of gap filling functions for motion capture marker data

## Description
Collection of several functions for gap filling and recovering of motion capture marker data

## Installation
GapFill can be installed from [PyPI](https://pypi.org/project/gapfill/) using ```pip``` on Python>=3.7.

```bash
pip install gapfill
```

## Usage
Rows of ndarray values for marker coordinates should be filled with numpy.nan for occluded (blocked) frames.
```python
import gapfill as gf

# numpy is required in order to provide necessary markers' coordinate values
import numpy as np

# 'fill_marker_gap_interp()' function will update the given ndarray by filling its gaps using bspline interpolation
# 'tgt_mkr_pos0': a 2D (n, 3) ndarray of a target marker position to fill the gaps
# 'n' is the total number of frames
tgt_mkr_pos0 = np.array((n, 3), dtype=np.float32)
# 'ret0': either True or False, True if there is any frame updated, False if there is no frame updated
# 'updated_frs_mask0': a boolean ndarray to indicate which frames are updated
ret0, updated_frs_mask0 = gf.fill_marker_gap_interp(tgt_mkr_pos0)

# 'fill_marker_gap_pattern()' function will update the given ndarray by filling its gaps using a donor marker
# 'tgt_mkr_pos1': a 2D (n, 3) ndarray of a target marker position to fill the gaps
# 'n' is the total number of frames
tgt_mkr_pos1 = np.array((n, 3), dtype=np.float32)
# 'dnr_mkr_pos': a 2D (n, 3) ndarray of a donor marker position
# 'n' is the total number of frames
dnr_mkr_pos = np.array((n, 3), dtype=np.float32)
# 'ret1': either True or False, True if there is any frame updated, False if there is no frame updated
# 'updated_frs_mask1': a boolean ndarray to indicate which frames are updated
ret1, updated_frs_mask1 = gf.fill_marker_gap_pattern(tgt_mkr_pos1, dnr_mkr_pos)

# 'fill_marker_gap_rbt()' function will update the given ndarray by filling its gaps using a cluster of 3 markers
# 'tgt_mkr_pos2': a 2D (n, 3) ndarray of a target marker position to fill the gaps
# 'n' is the total number of frames
tgt_mkr_pos2 = np.array((n, 3), dtype=np.float32)
# 'cl_mkr_pos': a 3D (m, n, 3) ndarray of the cluster markers
# 'm' (at least 3) is the number of markers, and 'n' is the total number of frames
cl_mkr_pos = np.array((m, n, 3), dtype=np.float32)
# 'ret2': either True or False, True if there is any frame updated, False if there is no frame updated
# 'updated_frs_mask2': a boolean ndarray to indicate which frames are updated
ret2, updated_frs_mask2 = gf.fill_marker_gap_rbt(tgt_mkr_pos2, cl_mkr_pos)
```
## Dependencies
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)

## References
- [Smolka, J. and Lukasik, E., 2016, July. "The rigid body gap filling algorithm". In 2016 9th International Conference on Human System Interactions (HSI) (pp. 337-343). IEEE.](https://doi.org/10.1109/HSI.2016.7529654)
- [Wikipedia: "Kabsch algorithm"](https://en.wikipedia.org/wiki/Kabsch_algorithm)
- [Kwon3D: "Computation of the Rotation Matrix"](http://www.kwon3d.com/theory/jkinem/rotmat.html)
- [Vicon: "What Gap Filling Algorithms are used Nexus 2?"](http://www.vicon.com/support/faqs/?q=what-gap-filling-algorithms-are-used-nexus-2)
- [Qualisys: "Featuring the Trajectory Editor in QTM"](https://www.qualisys.com/webinars/viewing-gap-filling-and-smoothing-data-with-the-trajectory-editor/)

## How to cite this work

Here is a suggestion to cite this GitHub repository:

> Jung, M. K. (2020) GapFill: A Python module of gap filling functions for motion capture marker data. GitHub repository, <https://github.com/mkjung99/gapfill>.

## License
[MIT](https://choosealicense.com/licenses/mit/)
