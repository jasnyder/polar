import torch

def potential_nematic(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, Gi):
        S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
        S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)
        S4 = torch.sum(torch.cross(qi, Gi, dim=2) * torch.cross(qi, Gi, dim=2), dim=2)

        # Combination must be symmetric to abide by Newton's 3rd Law
        lam = (lam_i + lam_j) / 2

        # Calculate semi-nematic potential
        S = S1 + lam[:, :, 1] * torch.abs(S2) + lam[:, :, 2] * torch.abs(S3) + lam[:, :, 3] * S4
        Vij = torch.exp(-d) - S * torch.exp(-d / 5)
        return Vij


def potential_vectorial(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, Gi):
        S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
        S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)
        S4 = torch.sum(torch.cross(qi, Gi, dim=2) * torch.cross(qi, Gi, dim=2), dim=2)

        # Combination must be symmetric to abide by Newton's 3rd Law
        lam = (lam_i + lam_j) / 2

        # Calculate semi-nematic potential
        S = S1 + lam[:, :, 1] * S2 + lam[:, :, 2] * S3 + lam[:, :, 3] * S4
        Vij = torch.exp(-d) - S * torch.exp(-d / 5)
        return Vij