"""AI_SD, four-direction reference-corrected Gumbel rho, SSIM, MAE and squared LNCC.

Rho: raw-selected direction, common-valid ROI fits, regional mean locations.
LNCC: mean local r^2 in 5x5x5 neighborhoods, not squared mean correlation.
References: Cammin 2024 (10.1002/acm2.14453), Nakane 2015
(10.6009/jjrt.2015_JSRT_71.12.1165), Wang 2004 (10.1109/TIP.2003.819861),
Avants 2008 (10.1016/j.media.2007.06.004).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

DIRECTIONS = ('x', 'y', 'diag_pos', 'diag_neg')


def corrected(artifact, reference):
    if artifact is None or reference is None:
        return None
    return float(np.sqrt(max(artifact*artifact-reference*reference, 0.)))


def directional_location(patch, direction):
    plane = np.asarray(patch, dtype=np.float64)
    if direction == 'x':
        profiles = plane.T
    elif direction == 'y':
        profiles = plane
    elif direction in ('diag_pos', 'diag_neg'):
        if direction == 'diag_neg':
            plane = np.fliplr(plane)
        profiles = [np.diagonal(plane, offset=k)
                    for k in range(-plane.shape[0]+1, plane.shape[1])]
    else:
        raise ValueError(direction)
    values = np.sort([float(np.abs(np.diff(p)).max()) for p in profiles if len(p) >= 2])
    if len(values) < 5 or np.ptp(values) <= 1e-8:
        return None, len(values)
    reduced = -np.log(-np.log(np.arange(1, len(values)+1)/(len(values)+1.)))
    slope, intercept = np.polyfit(values, reduced, 1)
    location = -intercept/slope
    return (float(location) if np.isfinite(location) and slope > 0 else None), len(values)


def artifact_metrics(images, rois):
    """Select direction from Input; retain common valid fits across all methods."""
    methods = list(images)
    fits = {d: {region: [] for region in ('artifact', 'reference')} for d in DIRECTIONS}
    sigmas = {m: {region: [] for region in ('artifact', 'reference')} for m in methods}
    for roi in rois:
        x0, x1, y0, y1, z = [int(roi[k]) for k in
                            ('x_start', 'x_stop', 'y_start', 'y_stop', 'slice_z')]
        patches = {m: a[x0:x1, y0:y1, z] for m, a in images.items()}
        for method, patch in patches.items():
            sigmas[method][roi['region']].append(float(patch.std(ddof=1, dtype=np.float64)))
        for d in DIRECTIONS:
            fits[d][roi['region']].append(
                {m: directional_location(patch, d)[0] for m, patch in patches.items()})

    def regional(method, direction, common):
        result = {}
        for region, groups in fits[direction].items():
            valid = [g[method] for g in groups if g[method] is not None
                     and (not common or all(v is not None for v in g.values()))]
            result['lambda_'+region] = float(np.mean(valid)) if valid else None
            result[region+'_rho_roi_count'] = len(valid)
        result['rho'] = corrected(result['lambda_artifact'], result['lambda_reference'])
        return result

    raw_scores = {d: regional('Input', d, False) for d in DIRECTIONS}
    valid = [d for d in DIRECTIONS if raw_scores[d]['rho'] is not None]
    direction = max(valid, key=lambda d: raw_scores[d]['rho']) if valid else None
    result = {}
    for method in methods:
        row = regional(method, direction, True) if direction else dict(
            lambda_artifact=None, lambda_reference=None, rho=None,
            artifact_rho_roi_count=0, reference_rho_roi_count=0)
        row['direction'] = direction
        for region in ('artifact', 'reference'):
            values = sigmas[method][region]
            row['sigma_'+region] = float(np.mean(values)) if values else None
            row[region+'_roi_count'] = len(values)
        row['ai_sd'] = corrected(row['sigma_artifact'], row['sigma_reference'])
        result[method] = row
    return result, raw_scores


def make_body_mask(raw):
    initial = raw > -500
    body = ndi.binary_closing(initial, iterations=2, border_value=1)
    return ndi.binary_fill_holes(body) | initial


def preservation_support(body, artifact, spacing):
    # Keep the original EDT convention, including its behavior for empty masks.
    expanded = ndi.distance_transform_edt(~(artifact & body), sampling=spacing) <= 5.
    return body & ~expanded


def masked_ssim(raw, output, mask):
    total, count = 0., 0
    for z in range(raw.shape[2]):
        local = mask[:, :, z]
        if not local.any():
            continue
        _, values = structural_similarity(raw[:, :, z], output[:, :, z],
            data_range=2000, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, full=True)
        total += float(values[local].sum())
        count += int(local.sum())
    return total/count if count else None


def squared_lncc(raw, output, support):
    a, b = np.asarray(raw, dtype=np.float64), np.asarray(output, dtype=np.float64)
    mean = lambda x: ndi.uniform_filter(x, (5, 5, 5), mode='reflect')
    ma, mb = mean(a), mean(b)
    va, vb = np.maximum(mean(a*a)-ma*ma, 0), np.maximum(mean(b*b)-mb*mb, 0)
    covariance = mean(a*b)-ma*mb
    valid = support & (va > 1e-8) & (vb > 1e-8)
    if not valid.any():
        return None, 0
    cc = np.clip(covariance[valid]/np.sqrt(va[valid]*vb[valid]), -1, 1)
    return float((cc*cc).mean()), int(valid.sum())


def preservation_metrics(raw, output, support):
    if not support.any():
        raise ValueError('Empty preservation support')
    lncc, count = squared_lncc(raw, output, support)
    return dict(outside_mae=float(np.abs(output[support]-raw[support]).mean(dtype=np.float64)),
                outside_ssim=masked_ssim(raw, output, support), outside_lncc=lncc,
                outside_voxels=int(support.sum()), lncc_valid_voxels=count)
