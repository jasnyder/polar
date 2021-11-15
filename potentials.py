import torch


def potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) *
                   torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) *
                   torch.cross(qj, dx, dim=2), dim=2)

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + \
        lam[:, :, 2] * S2 + lam[:, :, 3] * S3
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


def potential_nematic(x, d, dx, lam_i, lam_j, pi, pj, qi, qj):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.abs(torch.sum(torch.cross(pi, qi, dim=2)
                   * torch.cross(pj, qj, dim=2), dim=2))
    S3 = torch.abs(torch.sum(torch.cross(qi, dx, dim=2)
                   * torch.cross(qj, dx, dim=2), dim=2))

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + \
        lam[:, :, 2] * S2 + lam[:, :, 3] * S3
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


def potential_wnt_nematic(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, wnt_cells, wnt_range):
    # displacements from each cell to each WNT-producing cell. shape: n_cells x n_wnt_cells x 3
    G = x[:, None, :] - x[wnt_cells]
    GG = G.clone()
    Gnorm = torch.sqrt(torch.sum(G**2, dim=2))
    GG[Gnorm > wnt_range] = 0
    GG = torch.nn.functional.normalize(GG, dim=2, eps=1e-6)

    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.abs(torch.sum(torch.cross(pi, qi, dim=2)
                   * torch.cross(pj, qj, dim=2), dim=2))
    S3 = torch.abs(torch.sum(torch.cross(qi, dx, dim=2)
                   * torch.cross(qj, dx, dim=2), dim=2))
    S4 = torch.sum(torch.sum(torch.cross(qi[:, 0, :][:, None, :].expand(
        qi.shape[0], G.shape[1], 3), GG, dim=2)**2, dim=2), dim=1)

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + lam[:, :, 2] * S2 + lam[:, :,
                                                                   3] * S3 + lam[:, :, 4] * S4[:, None].expand(S4.shape[0], lam.shape[1])
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


def potential_wnt(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, wnt_cells, wnt_range):
    # displacements from each cell to each WNT-producing cell
    G = x[:, None, :] - x[wnt_cells]
    # displacements from each cell to each WNT-producing cell. shape: n_cells x n_wnt_cells x 3
    G = x[:, None, :] - x[wnt_cells]
    GG = G.clone()
    Gnorm = torch.sqrt(torch.sum(G**2, dim=2))
    GG[Gnorm > wnt_range] = 0
    GG = torch.nn.functional.normalize(GG, dim=2, eps=1e-6)

    S1 = torch.sum(torch.cross(pj, dx, dim=2) *
                   torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) *
                   torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) *
                   torch.cross(qj, dx, dim=2), dim=2)
    S4 = torch.sum(torch.sum(torch.cross(qi[:, 0, :][:, None, :].expand(
        qi.shape[0], G.shape[1], 3), G, dim=2)**2, dim=2), dim=1)

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + lam[:, :, 2] * \
        S2 + lam[:, :, 3] * S3 + lam[:, :, 4] * S4
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij
