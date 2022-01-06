import torch


def potential_nematic(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, Gi):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.abs(torch.sum(torch.cross(pi, qi, dim=2) *
                   torch.cross(pj, qj, dim=2), dim=2))
    S3 = torch.abs(torch.sum(torch.cross(qi, dx, dim=2) *
                   torch.cross(qj, dx, dim=2), dim=2))
    S4 = torch.sum(torch.cross(qi, Gi, dim=2) *
                   torch.cross(qi, Gi, dim=2), dim=2)

    # Combination must be symmetric to abide by Newton's 3rd Law
    lam = (lam_i + lam_j) / 2
    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    # Calculate semi-nematic potential
    S = lam[:, :, 0] + lam[:, :, 1] * S1 + \
        lam[:, :, 2] * S2 + lam[:, :, 3] * S3 + lam[:, :, 4] * S4
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


def potential_nematic_reweight(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, Gi):
    """
    Implements the nematic potential and re-weights lambda at cells whose WNT is below threshold

    to detect which cells have WNT level below threshold, check the norm of Gi
    """
    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.abs(torch.sum(torch.cross(pi, qi, dim=2) *
                   torch.cross(pj, qj, dim=2), dim=2))
    S3 = torch.abs(torch.sum(torch.cross(qi, dx, dim=2) *
                   torch.cross(qj, dx, dim=2), dim=2))
    S4 = torch.sum(torch.cross(qi, Gi, dim=2) *
                   torch.cross(qi, Gi, dim=2), dim=2)

    # Combination must be symmetric to abide by Newton's 3rd Law
    lam = (lam_i + lam_j) / 2
    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]
    reweight_mask = ((Gi**2).sum(axis=2).mean(axis=1) < 0.999)
    lam[reweight_mask, :, :] /= lam[reweight_mask, :, :-1].sum(axis=-1)[:, :, None].expand(reweight_mask.sum(), lam.shape[1], lam.shape[2])

    # Calculate semi-nematic potential
    S = lam[:, :, 0] + lam[:, :, 1] * S1 + \
        lam[:, :, 2] * S2 + lam[:, :, 3] * S3 + lam[:, :, 4] * S4
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


def potential_vectorial(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, Gi):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) *
                   torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) *
                   torch.cross(qj, dx, dim=2), dim=2)
    S4 = torch.sum(torch.cross(qi, Gi, dim=2) *
                   torch.cross(qi, Gi, dim=2), dim=2)

    # Combination must be symmetric to abide by Newton's 3rd Law
    lam = (lam_i + lam_j) / 2
    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    # Calculate semi-nematic potential
    S = lam[:, :, 0] + lam[:, :, 1] * S1 + \
        lam[:, :, 2] * S2 + lam[:, :, 3] * S3 + lam[:, :, 4] * S4
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij

