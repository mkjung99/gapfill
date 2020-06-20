"""
MIT License

Copyright (c) 2020 Moon Ki Jung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__author__ = "Moon Ki Jung, https://github.com/mkjung99/gapfill"
__version__ = "0.0.5"

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#%%
def recover_marker_rel(tgt_mkr_pos, cl_mkr_pos, msg=False):
    """
    Recover the trajectory of a marker using the relation between a group (cluster) of markers.
    
    The number of cluster markers is fixed as 3.
    This function extrapolates the target marker values if the cluster markers are available.
    
    First cluster marker (cl_mkr_pos[0]) will be used as the origin of the LCS(Local Coordinate System).
    Second cluster marker (cl_mkr_pos[1]) will be used in order to determine the X axis of the LCS.
    Third cluster marker (cl_mkr_pos[2]) will be used in order to determine the XY plane of the LCS.
    
    Parameters
    ----------
    tgt_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Target marker coordinates. Occluded(blocked) frame values should be filled with nan.
    cl_mkr_pos : ndarray (m, n, 3), where m(fixed as 3) is the number of cluster markers and n is the total number of frames.
        Cluster marker coordinates. Occluded(blocked) frame values should be filled with nan.
    msg : bool, optional
        Whether to show messages or not. The default is False.
    
    Returns
    -------
    bool
        True of False.
    ndarray or None
        Boolean ndarray (n, 3) for updated frames, where n is the total number of frames.
    
    References
    ----------
    .. [1] https://www.qualisys.com/webinars/viewing-gap-filling-and-smoothing-data-with-the-trajectory-editor/
    
    """
    input_dtype = type(tgt_mkr_pos[0,0])
    n_total_frs = tgt_mkr_pos.shape[0]
    tgt_mkr_valid_mask = ~np.any(np.isnan(tgt_mkr_pos), axis=1)
    n_tgt_mkr_valid_frs = np.count_nonzero(tgt_mkr_valid_mask)
    if n_tgt_mkr_valid_frs == 0:
        if msg: print("Skipped: no valid target marker frame!")
        return False, None
    if n_tgt_mkr_valid_frs == n_total_frs:
        if msg: print("Skipped: all target marker frames valid!")
        return False, None
    cl_mkr_valid_mask = np.all(~np.any(np.isnan(cl_mkr_pos), axis=2), axis=0)
    all_mkr_valid_mask = np.logical_and(cl_mkr_valid_mask, tgt_mkr_valid_mask)
    if not np.any(all_mkr_valid_mask):
        if msg: print("Skipped: no common valid frame among markers!")
        return False, None
    cl_mkr_only_valid_mask = np.logical_and(cl_mkr_valid_mask, np.logical_not(tgt_mkr_valid_mask))
    if not np.any(cl_mkr_only_valid_mask):
        if msg: print("Skipped: cluster markers not helpful!")
        return False, None
    all_mkr_valid_frs = np.where(all_mkr_valid_mask)[0]
    cl_mkr_only_valid_frs = np.where(cl_mkr_only_valid_mask)[0]
    p0 = cl_mkr_pos[0]
    p1 = cl_mkr_pos[1]
    p2 = cl_mkr_pos[2]
    vec0 = p1-p0
    vec1 = p2-p0
    vec0_norm = np.linalg.norm(vec0, axis=1, keepdims=True)
    vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
    vec0_unit = np.divide(vec0, vec0_norm, where=(vec0_norm!=0))
    vec1_unit = np.divide(vec1, vec1_norm, where=(vec1_norm!=0))
    vec2 = np.cross(vec0_unit, vec1_unit)
    vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
    vec2_unit = np.divide(vec2, vec2_norm, where=(vec2_norm!=0))
    vec_z = vec2_unit
    vec_x = vec0_unit
    vec_y = np.cross(vec_z, vec_x)
    mat_rot = np.array([vec_x.T, vec_y.T, vec_z.T]).T
    tgt_mkr_pos_rel = np.einsum('ij,ijk->ik', (tgt_mkr_pos-p0)[all_mkr_valid_mask], mat_rot[all_mkr_valid_mask])
    tgt_mkr_pos_recovered = np.zeros((cl_mkr_only_valid_frs.size, 3), dtype=input_dtype)
    for idx, fr in np.ndenumerate(cl_mkr_only_valid_frs):
        search_idx = np.searchsorted(all_mkr_valid_frs, fr)
        if search_idx>=all_mkr_valid_frs.shape[0] or search_idx==0:
            tgt_pos_rel_idx = (np.abs(all_mkr_valid_frs-fr)).argmin()
            tgt_pos_rel = tgt_mkr_pos_rel[tgt_pos_rel_idx]
        else:
            idx1 = search_idx
            idx0 = search_idx-1
            fr1 = all_mkr_valid_frs[idx1]
            fr0 = all_mkr_valid_frs[idx0]
            a = input_dtype(fr-fr0)
            b = input_dtype(fr1-fr)
            tgt_pos_rel = (b*tgt_mkr_pos_rel[idx0]+a*tgt_mkr_pos_rel[idx1])/(a+b)
        tgt_mkr_pos_recovered[idx] = p0[fr]+np.dot(mat_rot[fr], tgt_pos_rel)
    tgt_mkr_pos[cl_mkr_only_valid_mask] = tgt_mkr_pos_recovered
    if msg: print("Updated.")
    return True, cl_mkr_only_valid_mask

def recover_marker_rbt(tgt_mkr_pos, cl_mkr_pos, msg=False):
    """
    Recover the trajectory of a marker by rbt(rigid body transformation) using a group (cluster) markers.
    
    The number of cluster markers is fixed as 3.
    This function extrapolates the target marker values if the cluster markers are available.
    The order of the cluster markers will be sorted according to their relative distances from the target marker.
    
    Parameters
    ----------
    tgt_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Target marker coordinates. Occluded(blocked) frame values should be filled with nan.
    cl_mkr_pos : ndarray (m, n, 3), where m(fixed as 3) is the number of cluster markers and n is the total number of frames.
        Cluster marker coordinates. Occluded(blocked) frame values should be filled with nan.
    msg : bool, optional
        Whether to show messages or not. The default is False.
    
    Returns
    -------
    bool
        True of False.
    ndarray or None
        Boolean ndarray (n, 3) for updated frames, where n is the total number of frames.
    
    References
    ----------
    .. [1] https://doi.org/10.1109/HSI.2016.7529654
    
    """
    input_dtype = type(tgt_mkr_pos[0,0])
    n_total_frs = tgt_mkr_pos.shape[0]
    tgt_mkr_valid_mask = ~np.any(np.isnan(tgt_mkr_pos), axis=1)
    n_tgt_mkr_valid_frs = np.count_nonzero(tgt_mkr_valid_mask)
    if n_tgt_mkr_valid_frs == 0:
        if msg: print("Skipped: no valid target marker frame!")
        return False, None
    if n_tgt_mkr_valid_frs == n_total_frs:
        if msg: print("Skipped: all target marker frames valid!")
        return False, None
    cl_mkr_valid_mask = np.all(~np.any(np.isnan(cl_mkr_pos), axis=2), axis=0)
    all_mkr_valid_mask = np.logical_and(cl_mkr_valid_mask, tgt_mkr_valid_mask)
    if not np.any(all_mkr_valid_mask):
        if msg: print("Skipped: no common valid frame among markers!")
        return False, None
    cl_mkr_only_valid_mask = np.logical_and(cl_mkr_valid_mask, np.logical_not(tgt_mkr_valid_mask))
    if not np.any(cl_mkr_only_valid_mask):
        if msg: print("Skipped: cluster markers not helpful!")
        return False, None
    all_mkr_valid_frs = np.where(all_mkr_valid_mask)[0]
    cl_mkr_only_valid_frs = np.where(cl_mkr_only_valid_mask)[0]
    dict_cl_mkr_dist = {}
    for i in range(cl_mkr_pos.shape[0]):
        vec_diff = cl_mkr_pos[i]-tgt_mkr_pos
        dict_cl_mkr_dist.update({i: np.nanmean(np.linalg.norm(vec_diff, axis=1))})
    cl_mkr_dist_sorted = sorted(dict_cl_mkr_dist.items(), key=lambda kv: kv[1])
    p0 = cl_mkr_pos[cl_mkr_dist_sorted[0][0]]
    p1 = cl_mkr_pos[cl_mkr_dist_sorted[1][0]]
    p2 = cl_mkr_pos[cl_mkr_dist_sorted[2][0]]
    p3 = tgt_mkr_pos
    vec0 = p1-p0
    vec1 = p2-p0
    vec0_norm = np.linalg.norm(vec0, axis=1, keepdims=True)
    vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
    vec0_unit = np.divide(vec0, vec0_norm, where=(vec0_norm!=0))
    vec1_unit = np.divide(vec1, vec1_norm, where=(vec1_norm!=0))
    vec2 = np.cross(vec0_unit, vec1_unit)
    vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
    vec2_unit = np.divide(vec2, vec2_norm, where=(vec2_norm!=0))
    vec3 = p3-p0
    vec_z = vec2_unit
    vec_x = vec0_unit
    vec_y = np.cross(vec_z, vec_x)
    mat_rot = np.array([vec_x.T, vec_y.T, vec_z.T]).T
    for idx, fr in np.ndenumerate(cl_mkr_only_valid_frs):
        search_idx = np.searchsorted(all_mkr_valid_frs, fr)
        if search_idx == 0:
            fr0 = all_mkr_valid_frs[0]
            rot_fr0_to_fr = np.dot(mat_rot[fr], mat_rot[fr0].T)
            vt_fr0 = np.dot(rot_fr0_to_fr, vec3[fr0])
            vc = vt_fr0
        elif search_idx >= all_mkr_valid_frs.shape[0]:
            fr1 = all_mkr_valid_frs[all_mkr_valid_frs.shape[0]-1]
            rot_fr1_to_fr = np.dot(mat_rot[fr], mat_rot[fr1].T)
            vt_fr1 = np.dot(rot_fr1_to_fr, vec3[fr1])
            vc = vt_fr1
        else:
            fr0 = all_mkr_valid_frs[search_idx-1]
            fr1 = all_mkr_valid_frs[search_idx]
            rot_fr0_to_fr = np.dot(mat_rot[fr], mat_rot[fr0].T)
            rot_fr1_to_fr = np.dot(mat_rot[fr], mat_rot[fr1].T)
            vt_fr0 = np.dot(rot_fr0_to_fr, vec3[fr0])
            vt_fr1 = np.dot(rot_fr1_to_fr, vec3[fr1])
            a = input_dtype(fr-fr0)
            b = input_dtype(fr1-fr)
            vc = (b*vt_fr0+a*vt_fr1)/(a+b)
        tgt_mkr_pos[fr] = p0[fr]+vc
    if msg: print("Updated.")
    return True, cl_mkr_only_valid_mask

def fill_marker_gap_rbt(tgt_mkr_pos, cl_mkr_pos, msg=False):
    """
    Recover the trajectory of a marker by rbt(rigid body transformation) using a group (cluster) markers.
    
    Parameters
    ----------
    tgt_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Target marker coordinates. Occluded(blocked) frame values should be filled with nan.
    cl_mkr_pos : ndarray (m, n, 3), where m is the number of cluster markers and n is the total number of frames.
        Cluster marker coordinates. Occluded(blocked) frame values should be filled with nan.
    msg : bool, optional
        Whether to show messages or not. The default is False.
    
    Returns
    -------
    bool
        True of False.
    ndarray or None
        Boolean ndarray (n, 3) for updated frames, where n is the total number of frames.
    
    References
    ----------
    .. [1] http://www.vicon.com/support/faqs/?q=what-gap-filling-algorithms-are-used-nexus-2
    .. [2] http://www.kwon3d.com/theory/jkinem/rotmat.html
    .. [3] https://en.wikipedia.org/wiki/Kabsch_algorithm
    .. [4] https://doi.org/10.1109/TPAMI.1987.4767965
    
    """
    def RBT(A, B):
        Ac = A.mean(axis=0)
        Bc = B.mean(axis=0)
        C = np.dot((B-Bc).T, (A-Ac))
        U, S, Vt = np.linalg.svd(C)
        R = np.dot(U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt))
        t = Bc-np.dot(R, Ac)
        err_vec = np.dot(R, A.T).T+t-B
        err_norm = np.linalg.norm(err_vec, axis=1)
        mean_err_norm = np.mean(err_norm)
        return R, t, err_vec, err_norm, mean_err_norm
    input_dtype = type(tgt_mkr_pos[0,0])
    n_total_frs = tgt_mkr_pos.shape[0]
    tgt_mkr_valid_mask = ~np.any(np.isnan(tgt_mkr_pos), axis=1)
    n_tgt_mkr_valid_frs = np.count_nonzero(tgt_mkr_valid_mask)
    if n_tgt_mkr_valid_frs == 0:
        if msg: print("Skipped: no valid target marker frame!")
        return False, None
    if n_tgt_mkr_valid_frs == n_total_frs:
        if msg: print("Skipped: all target marker frames valid!")
        return False, None
    cl_mkr_valid_mask = np.all(~np.any(np.isnan(cl_mkr_pos), axis=2), axis=0)
    all_mkr_valid_mask = np.logical_and(cl_mkr_valid_mask, tgt_mkr_valid_mask)
    if not np.any(all_mkr_valid_mask):
        if msg: print("Skipped: no common valid frame among markers!")
        return False, None
    cl_mkr_only_valid_mask = np.logical_and(cl_mkr_valid_mask, np.logical_not(tgt_mkr_valid_mask))
    if not np.any(cl_mkr_only_valid_mask):
        if msg: print("Skipped: cluster markers not helpful!")
        return False, None
    all_mkr_valid_frs = np.where(all_mkr_valid_mask)[0]
    cl_mkr_only_valid_frs = np.where(cl_mkr_only_valid_mask)[0]
    tgt_mkr_updated_mask = cl_mkr_only_valid_mask.copy()
    b_updated = False
    for idx, fr in np.ndenumerate(cl_mkr_only_valid_frs):
        search_idx = np.searchsorted(all_mkr_valid_frs, fr)
        if search_idx == 0:
            fr0 = all_mkr_valid_frs[0]
            fr1 = all_mkr_valid_frs[1]
        elif search_idx >= all_mkr_valid_frs.shape[0]:
            fr0 = all_mkr_valid_frs[all_mkr_valid_frs.shape[0]-2]
            fr1 = all_mkr_valid_frs[all_mkr_valid_frs.shape[0]-1]
        else:
            fr0 = all_mkr_valid_frs[search_idx-1]
            fr1 = all_mkr_valid_frs[search_idx]
        if fr <= fr0 or fr >= fr1:
            tgt_mkr_updated_mask[fr] = False
            continue
        if ~cl_mkr_valid_mask[fr0] or ~cl_mkr_valid_mask[fr1]:
            tgt_mkr_updated_mask[fr] = False
            continue
        if np.any(~cl_mkr_valid_mask[fr0:fr1+1]):
            tgt_mkr_updated_mask[fr] = False
            continue
        cl_mkr_pos_fr0 = np.zeros((cl_mkr_pos.shape[0], 3), dtype=input_dtype)
        cl_mkr_pos_fr1 = np.zeros((cl_mkr_pos.shape[0], 3), dtype=input_dtype)
        cl_mkr_pos_fr = np.zeros((cl_mkr_pos.shape[0], 3), dtype=input_dtype)
        for i in range(cl_mkr_pos.shape[0]):
            cl_mkr_pos_fr0[i,:] = cl_mkr_pos[i][fr0,:]
            cl_mkr_pos_fr1[i,:] = cl_mkr_pos[i][fr1,:]
            cl_mkr_pos_fr[i,:] = cl_mkr_pos[i][fr,:]
        rot_fr0, trans_fr0, _, _, _ = RBT(cl_mkr_pos_fr0, cl_mkr_pos_fr)
        rot_fr1, trans_fr1, _, _, _ = RBT(cl_mkr_pos_fr1, cl_mkr_pos_fr)
        tgt_mkr_pos_fr_fr0 = np.dot(rot_fr0, tgt_mkr_pos[fr0])+trans_fr0
        tgt_mkr_pos_fr_fr1 = np.dot(rot_fr1, tgt_mkr_pos[fr1])+trans_fr1
        tgt_mkr_pos[fr] = (tgt_mkr_pos_fr_fr1-tgt_mkr_pos_fr_fr0)*input_dtype(fr-fr0)/input_dtype(fr1-fr0)+tgt_mkr_pos_fr_fr0
        b_updated = True        
    if b_updated:
        if msg: print("Updated.")
        return True, tgt_mkr_updated_mask
    else:
        if msg: print("Skipped.")
        return False, None
    
def fill_marker_gap_pattern(tgt_mkr_pos, dnr_mkr_pos, search_span_offset=5, min_needed_frs=10, msg=False):
    """
    Fill the gaps in a given target marker coordinates using the donor marker coordinates by linear interpolation.
    
    Parameters
    ----------
    tgt_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Target marker coordinates. Occluded(blocked) frame values should be filled with nan.        
    dnr_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Donor marker coordinates. Occluded(blocked) frame values should be filled with nan.
    search_span_offset : int, optional
        Offset for backward and forward search spans. The default is 5.
    min_needed_frs : int, optional
        Minimum required valid frames in both search spans. The default is 10.
    msg : bool, optional
        Whether to show messages or not. The default is False.

    Returns
    -------
    bool
        True or False.
    ndarray or None
        Boolean ndarray (n, 3) for updated frames, where n is the total number of frames.
        
    References
    ----------
    .. [1] http://www.vicon.com/support/faqs/?q=what-gap-filling-algorithms-are-used-nexus-2
    
    """
    input_dtype = type(tgt_mkr_pos[0,0])
    n_total_frs = tgt_mkr_pos.shape[0]
    tgt_mkr_valid_mask = ~np.any(np.isnan(tgt_mkr_pos), axis=1)
    tgt_mkr_invalid_mask = ~tgt_mkr_valid_mask
    n_tgt_mkr_valid_frs = np.count_nonzero(tgt_mkr_valid_mask)
    if n_tgt_mkr_valid_frs==0 or n_tgt_mkr_valid_frs==n_total_frs:
        if msg: print("Skipped.")
        return False, None
    dnr_mkr_valid_mask = ~np.any(np.isnan(dnr_mkr_pos), axis=1)
    if not np.any(dnr_mkr_valid_mask):
        if msg: print("Skipped.")
        return False, None    
    both_mkr_valid_mask = np.logical_and(tgt_mkr_valid_mask, dnr_mkr_valid_mask)
    if not np.any(both_mkr_valid_mask):
        if msg: print("Skipped.")
        return False, None
    tgt_mkr_invalid_frs = np.where(tgt_mkr_invalid_mask)[0]
    tgt_mkr_invalid_gaps = np.split(tgt_mkr_invalid_frs, np.where(np.diff(tgt_mkr_invalid_frs)!=1)[0]+1)
    tgt_mkr_updated_mask = tgt_mkr_invalid_mask.copy()
    b_updated = False
    for gap in tgt_mkr_invalid_gaps:
        # Skip if gap size is zero
        if gap.size == 0:
            tgt_mkr_updated_mask[gap] = False
            continue
        # Skip if gap is either at the first or at the end of the entire frames.
        if gap.min()==0 or gap.max()==n_total_frs-1:
            tgt_mkr_updated_mask[gap] = False
            continue
        search_span = np.int(np.ceil(gap.size/2))+search_span_offset
        gap_near_tgt_mkr_valid_mask = np.zeros((n_total_frs,), dtype=bool)
        for i in range(gap.min()-1, gap.min()-1-search_span, -1):
            if i >= 0: gap_near_tgt_mkr_valid_mask[i]=True
        for i in range(gap.max()+1, gap.max()+1+search_span, 1):
            if i < n_total_frs: gap_near_tgt_mkr_valid_mask[i]=True
        gap_near_tgt_mkr_valid_mask = np.logical_and(gap_near_tgt_mkr_valid_mask, tgt_mkr_valid_mask)
        # Skip if total number of available target marker frames near the gap within search span is less then minimum required number.
        if np.sum(gap_near_tgt_mkr_valid_mask) < min_needed_frs:
            tgt_mkr_updated_mask[gap] = False
            continue
        # Skip if there is any invalid frame of the donor marker during the gap period.
        if np.any(~dnr_mkr_valid_mask[gap]):
            tgt_mkr_updated_mask[gap] = False
            continue
        gap_near_both_mkr_valid_mask = np.logical_and(gap_near_tgt_mkr_valid_mask, dnr_mkr_valid_mask)
        gap_near_both_mkr_valid_frs = np.where(gap_near_both_mkr_valid_mask)[0]
        for idx, fr in np.ndenumerate(gap):
            search_idx = np.searchsorted(gap_near_both_mkr_valid_frs, fr)
            if search_idx == 0:
                fr0 = gap_near_both_mkr_valid_frs[0]
                fr1 = gap_near_both_mkr_valid_frs[1]
            elif search_idx >= gap_near_both_mkr_valid_frs.shape[0]:
                fr0 = gap_near_both_mkr_valid_frs[gap_near_both_mkr_valid_frs.shape[0]-2]
                fr1 = gap_near_both_mkr_valid_frs[gap_near_both_mkr_valid_frs.shape[0]-1]
            else:
                fr0 = gap_near_both_mkr_valid_frs[search_idx-1]
                fr1 = gap_near_both_mkr_valid_frs[search_idx]
            # Skip if the target marker frame fr is outside of range.
            if fr <= fr0 or fr >= fr1:
                tgt_mkr_updated_mask[fr] = False
                continue
            # Skip if the donor marker is invalid at either fr0 or fr1.
            if ~dnr_mkr_valid_mask[fr0] or ~dnr_mkr_valid_mask[fr1]:
                tgt_mkr_updated_mask[fr] = False
                continue
            v_tgt = (tgt_mkr_pos[fr1]-tgt_mkr_pos[fr0])*input_dtype(fr-fr0)/input_dtype(fr1-fr0)+tgt_mkr_pos[fr0]
            v_dnr = (dnr_mkr_pos[fr1]-dnr_mkr_pos[fr0])*input_dtype(fr-fr0)/input_dtype(fr1-fr0)+dnr_mkr_pos[fr0]
            tgt_mkr_pos[fr] = v_tgt-v_dnr+dnr_mkr_pos[fr]
            b_updated = True
    if b_updated:
        if msg: print("Updated.")
        return True, tgt_mkr_updated_mask
    else:
        if msg: print("Skipped.")
        return False, None
    
def fill_marker_gap_pattern2(tgt_mkr_pos, dnr_mkr_pos, msg=False):
    """
    Fill the gaps in a given target marker coordinates using the donor marker coordinates by linear interpolation.
    
    Parameters
    ----------
    tgt_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Target marker coordinates. Occluded(blocked) frame values should be filled with nan.        
    dnr_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Donor marker coordinates. Occluded(blocked) frame values should be filled with nan.
    msg : bool, optional
        Whether to show messages or not. The default is False.

    Returns
    -------
    bool
        True or False.
    ndarray or None
        Boolean ndarray (n, 3) for updated frames, where n is the total number of frames.

    Notes
    -----
    This function is less strict than fill_marker_gap_pattern(),
    because this function does not check search spans and minimum required valid frames around each gap.
    
    References
    ----------
    .. [1] http://www.vicon.com/support/faqs/?q=what-gap-filling-algorithms-are-used-nexus-2

    """
    input_dtype = type(tgt_mkr_pos[0,0])
    n_total_frs = tgt_mkr_pos.shape[0]
    tgt_mkr_valid_mask = ~np.any(np.isnan(tgt_mkr_pos), axis=1)
    tgt_mkr_invalid_mask = ~tgt_mkr_valid_mask
    n_tgt_mkr_valid_frs = np.count_nonzero(tgt_mkr_valid_mask)
    if n_tgt_mkr_valid_frs==0 or n_tgt_mkr_valid_frs==n_total_frs:
        if msg: print("Skipped.")
        return False, None
    dnr_mkr_valid_mask = ~np.any(np.isnan(dnr_mkr_pos), axis=1)
    if not np.any(dnr_mkr_valid_mask):
        if msg: print("Skipped.")
        return False, None    
    both_mkr_valid_mask = np.logical_and(tgt_mkr_valid_mask, dnr_mkr_valid_mask)
    if not np.any(both_mkr_valid_mask):
        if msg: print("Skipped.")
        return False, None
    tgt_mkr_invalid_frs = np.where(tgt_mkr_invalid_mask)[0]
    both_mkr_valid_frs = np.where(both_mkr_valid_mask)[0]
    tgt_mkr_updated_mask = tgt_mkr_invalid_mask.copy()
    b_updated = False
    for idx, fr in np.ndenumerate(tgt_mkr_invalid_frs):
        search_idx = np.searchsorted(both_mkr_valid_frs, fr)
        if search_idx == 0:
            fr0 = both_mkr_valid_frs[0]
            fr1 = both_mkr_valid_frs[1]
        elif search_idx >= both_mkr_valid_frs.shape[0]:
            fr0 = both_mkr_valid_frs[both_mkr_valid_frs.shape[0]-2]
            fr1 = both_mkr_valid_frs[both_mkr_valid_frs.shape[0]-1]
        else:
            fr0 = both_mkr_valid_frs[search_idx-1]
            fr1 = both_mkr_valid_frs[search_idx]
        if fr <= fr0 or fr >= fr1:
            tgt_mkr_updated_mask[fr] = False
            continue
        if ~dnr_mkr_valid_mask[fr0] or ~dnr_mkr_valid_mask[fr1]:
            tgt_mkr_updated_mask[fr] = False
            continue
        if np.any(~dnr_mkr_valid_mask[fr0:fr1+1]):
            tgt_mkr_updated_mask[fr] = False
            continue    
        v_tgt = (tgt_mkr_pos[fr1]-tgt_mkr_pos[fr0])*input_dtype(fr-fr0)/input_dtype(fr1-fr0)+tgt_mkr_pos[fr0]
        v_dnr = (dnr_mkr_pos[fr1]-dnr_mkr_pos[fr0])*input_dtype(fr-fr0)/input_dtype(fr1-fr0)+dnr_mkr_pos[fr0]
        tgt_mkr_pos[fr] = v_tgt-v_dnr+dnr_mkr_pos[fr]
        b_updated = True
    if b_updated:
        if msg: print("Updated.")
        return True, tgt_mkr_updated_mask
    else:
        if msg: print("Skipped.")
        return False, None

def fill_marker_gap_interp(tgt_mkr_pos, k=3, search_span_offset=5, min_needed_frs=10, msg=False):
    """
    Fill the gaps in a given target marker coordinates using scipy.interpolate.InterpolatedUnivariateSpline function.

    Parameters
    ----------
    tgt_mkr_pos : ndarray (n, 3), where n is the total number of frames.
        Target marker coordinates. Occluded(blocked) frame values should be filled with nan.
    k : int, optional
        Degrees of smoothing spline. The default is 3.
    search_span_offset : int, optional
        Offset for backward and forward search spans. The default is 5.
    min_needed_frs : int, optional
        Minimum required valid frames in a search span. The default is 10.
    msg : bool, optional
        Whether to show messages or not. The default is False.

    Returns
    -------
    bool
        True or False.
    ndarray or None
        Boolean ndarray (n, 3) for updated frames, where n is the total number of frames.
        
    References
    ----------
    .. [1] http://www.vicon.com/support/faqs/?q=what-gap-filling-algorithms-are-used-nexus-2        

    """
    n_total_frs = tgt_mkr_pos.shape[0]
    tgt_mkr_valid_mask = ~np.any(np.isnan(tgt_mkr_pos), axis=1)
    tgt_mkr_invalid_mask = ~tgt_mkr_valid_mask
    n_tgt_mkr_valid_frs = np.count_nonzero(tgt_mkr_valid_mask)
    if n_tgt_mkr_valid_frs==0 or n_tgt_mkr_valid_frs==n_total_frs:
        if msg: print("Skipped.")
        return False, None
    tgt_mkr_invalid_frs = np.where(tgt_mkr_invalid_mask)[0]
    tgt_mkr_invalid_gaps = np.split(tgt_mkr_invalid_frs, np.where(np.diff(tgt_mkr_invalid_frs)!=1)[0]+1)
    tgt_mkr_updated_mask = tgt_mkr_invalid_mask.copy()
    b_updated = False
    for gap in tgt_mkr_invalid_gaps:
        if gap.size == 0:
            tgt_mkr_updated_mask[gap] = False
            continue
        if gap.min()==0 or gap.max()==n_total_frs-1:
            tgt_mkr_updated_mask[gap] = False
            continue
        search_span = np.int(np.ceil(gap.size/2))+search_span_offset
        itpl_cand_frs_mask = np.zeros((n_total_frs,), dtype=bool)
        for i in range(gap.min()-1, gap.min()-1-search_span, -1):
            if i >= 0: itpl_cand_frs_mask[i]=True
        for i in range(gap.max()+1, gap.max()+1+search_span, 1):
            if i < n_total_frs: itpl_cand_frs_mask[i]=True
        itpl_cand_frs_mask = np.logical_and(itpl_cand_frs_mask, tgt_mkr_valid_mask)
        if np.sum(itpl_cand_frs_mask) < min_needed_frs:
            tgt_mkr_updated_mask[gap] = False
            continue
        itpl_cand_frs = np.where(itpl_cand_frs_mask)[0]
        itpl_cand_pos = tgt_mkr_pos[itpl_cand_frs, :]
        fun_itpl_x = InterpolatedUnivariateSpline(itpl_cand_frs, itpl_cand_pos[:,0], k=k, ext='const')
        fun_itpl_y = InterpolatedUnivariateSpline(itpl_cand_frs, itpl_cand_pos[:,1], k=k, ext='const')
        fun_itpl_z = InterpolatedUnivariateSpline(itpl_cand_frs, itpl_cand_pos[:,2], k=k, ext='const')
        itpl_x = fun_itpl_x(gap)
        itpl_y = fun_itpl_y(gap)
        itpl_z = fun_itpl_z(gap)
        for idx, fr in enumerate(gap):
            tgt_mkr_pos[fr,0] = itpl_x[idx]
            tgt_mkr_pos[fr,1] = itpl_y[idx]
            tgt_mkr_pos[fr,2] = itpl_z[idx]
        b_updated = True            
    if b_updated:
        if msg: print("Updated.")
        return True, tgt_mkr_updated_mask
    else:
        if msg: print("Skipped.")
        return False, None