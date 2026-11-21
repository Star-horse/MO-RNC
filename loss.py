import torch
import torch.nn as nn

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1', lmmo_M=None, lmmo_V=None):
        super().__init__()
        self.distance_type = distance_type

        if distance_type == 'lmmo_knn' and (lmmo_M is None or lmmo_V is None):
            raise ValueError('lmmo_knn requires LMMO M and V.')
        if distance_type == 'yqy' and lmmo_V is None:
            raise ValueError('yqy requires LMMO V.')

        self.register_buffer(
            'lmmo_M',
            torch.empty(0) if lmmo_M is None
            else torch.as_tensor(lmmo_M, dtype=torch.float32),
        )
        self.register_buffer(
            'lmmo_V',
            torch.empty(0) if lmmo_V is None
            else torch.as_tensor(lmmo_V, dtype=torch.float32),
        )

    def forward(self, labels, inputs=None):
        if self.distance_type == 'l1':
            return torch.cdist(labels, labels, p=1)
        if self.distance_type == 'l2':
            return torch.cdist(labels, labels, p=2)
        if self.distance_type == 'lmmo_knn':
            encoded = inputs.to(self.lmmo_M.dtype) @ self.lmmo_M @ self.lmmo_V
            return torch.cdist(encoded, encoded, p=2).square()
        if self.distance_type == 'yqy':
            encoded = labels.to(self.lmmo_V.dtype) @ self.lmmo_V
            return torch.cdist(encoded, encoded, p=2).square()
        raise ValueError(f'Unknown label distance: {self.distance_type}.')

class RnCLoss(nn.Module):
    def __init__(
        self,
        temperature_alpha=1.0,
        temperature_beta=0.1,
        label_diff='l1',
        input_knn_k=None,
        lmmo_M=None,
        lmmo_V=None,
    ):
        super().__init__()
        if temperature_alpha < 0 or temperature_beta <= 0:
            raise ValueError(
                'temperature_alpha must be non-negative and '
                'temperature_beta must be positive.'
            )
        self.temperature_alpha = float(temperature_alpha)
        self.temperature_beta = float(temperature_beta)
        self.input_knn_k = input_knn_k
        self.label_diff_fn = LabelDifference(
            label_diff,
            lmmo_M=lmmo_M,
            lmmo_V=lmmo_V,
        )

    def _input_neighborhood_mask(self, inputs, n, device):
        if self.input_knn_k is None:
            return ~torch.eye(n, dtype=torch.bool, device=device)

        distances = torch.cdist(inputs.detach(), inputs.detach(), p=2)
        distances.fill_diagonal_(float('inf'))
        k = min(self.input_knn_k, n - 1)
        neighbors = distances.topk(k, dim=1, largest=False).indices
        mask = torch.zeros((n, n), dtype=torch.bool, device=inputs.device)
        mask.scatter_(1, neighbors, True)
        return mask.to(device)

    def forward(self, features, labels, inputs=None):
        if features.dim() == 3:
            batch_size = features.shape[0]
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)
            labels = (
                torch.cat([labels[:, 0], labels[:, 1]], dim=0)
                if labels.dim() == 3 else labels.repeat(2, 1)
            )
            if inputs is not None and inputs.shape[0] == batch_size:
                inputs = inputs.repeat((2,) + (1,) * (inputs.dim() - 1))

        n = features.shape[0]
        neighborhood_mask = self._input_neighborhood_mask(
            inputs, n, features.device
        )
        relation_diffs = self.label_diff_fn(labels, inputs=inputs)
        off_diagonal = ~torch.eye(
            n, dtype=torch.bool, device=features.device
        )
        mean_distance = relation_diffs[off_diagonal].view(n, n - 1).mean(1)
        temperatures = (
            self.temperature_alpha * mean_distance + self.temperature_beta
        ).detach()
        logits = (
            -torch.cdist(features, features, p=2) / temperatures[:, None]
        )
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = logits.exp()

        logits = logits[off_diagonal].view(n, n - 1)
        exp_logits = exp_logits[off_diagonal].view(n, n - 1)
        relation_diffs = relation_diffs[off_diagonal].view(n, n - 1)
        neighborhood_mask = neighborhood_mask[off_diagonal].view(n, n - 1)

        sorted_diffs, order = relation_diffs.sort(dim=1)
        sorted_logits = logits.gather(1, order)
        sorted_exp_logits = exp_logits.gather(1, order)
        sorted_neighbors = neighborhood_mask.gather(1, order)

        suffix_sums = (
            sorted_exp_logits.masked_fill(~sorted_neighbors, 0)
            .flip(1).cumsum(1).flip(1)
        )

        group_starts = torch.cat([
            torch.ones((n, 1), dtype=torch.bool, device=features.device),
            sorted_diffs[:, 1:] != sorted_diffs[:, :-1],
        ], dim=1)
        positions = torch.arange(
            n - 1, device=features.device
        ).unsqueeze(0).expand_as(sorted_diffs)
        group_first = torch.where(
            group_starts, positions, torch.zeros_like(positions)
        ).cummax(dim=1).values
        denominator = suffix_sums.gather(1, group_first).clamp_min(
            torch.finfo(exp_logits.dtype).tiny
        )

        log_prob = sorted_logits - denominator.log()
        loss = -log_prob[sorted_neighbors].sum()

        return loss / neighborhood_mask.sum()

class MoRnCLoss(nn.Module):

    def __init__(
        self,
        mahalanobis_matrix=None,
        global_distance='mahalanobis',
        local_weight=0.5,
        local_dims=2,
        temperature_alpha=1.0,
        temperature_beta=0.1,
        input_knn_k=None,
    ):
        super().__init__()
        if global_distance not in ('mahalanobis', 'l2'):
            raise ValueError("global_distance must be 'mahalanobis' or 'l2'.")
        if global_distance == 'mahalanobis':
            if mahalanobis_matrix is None:
                raise ValueError('mahalanobis_matrix is required for Mo-RNC.')
            matrix = torch.as_tensor(mahalanobis_matrix, dtype=torch.float32)
            if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError('mahalanobis_matrix must be a square matrix.')

            matrix = 0.5 * (matrix + matrix.T)
            if torch.linalg.eigvalsh(matrix).min() < -1e-5:
                raise ValueError(
                    'mahalanobis_matrix must be positive semidefinite.'
                )
        else:
            matrix = torch.empty(0, dtype=torch.float32)
        if local_dims < 1:
            raise ValueError('local_dims must be at least 1.')
        if temperature_alpha < 0 or temperature_beta <= 0:
            raise ValueError(
                'temperature_alpha must be non-negative and '
                'temperature_beta must be positive.'
            )

        self.register_buffer('mahalanobis_matrix', matrix)
        self.global_distance = global_distance
        self.local_weight = float(local_weight)
        self.local_dims = int(local_dims)
        self.temperature_alpha = float(temperature_alpha)
        self.temperature_beta = float(temperature_beta)
        self.input_knn_k = input_knn_k

        self.last_global_loss = None
        self.last_local_loss = None
        self.last_temperature = None
        self.last_local_dims = None

    @staticmethod
    def _expand_two_views(features, labels, inputs):
        if features.dim() != 3:
            return features, labels, inputs

        batch_size = features.shape[0]
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)
        labels = (
            torch.cat([labels[:, 0], labels[:, 1]], dim=0)
            if labels.dim() == 3 else labels.repeat(2, 1)
        )
        if inputs is not None and inputs.shape[0] == batch_size:
            inputs = inputs.repeat((2,) + (1,) * (inputs.dim() - 1))
        return features, labels, inputs

    def _input_neighborhood_mask(self, inputs, n, device):
        if self.input_knn_k is None:
            return ~torch.eye(n, dtype=torch.bool, device=device)
        if inputs is None:
            raise ValueError('inputs are required when input_knn_k is enabled.')

        distances = torch.cdist(inputs.detach(), inputs.detach(), p=2)
        distances.fill_diagonal_(float('inf'))
        k = min(self.input_knn_k, n - 1)
        neighbors = distances.topk(k, dim=1, largest=False).indices
        mask = torch.zeros((n, n), dtype=torch.bool, device=inputs.device)
        mask.scatter_(1, neighbors, True)
        return mask.to(device)

    def _mahalanobis_distances(self, labels):
        labels = labels.to(self.mahalanobis_matrix.dtype)
        differences = labels[:, None, :] - labels[None, :, :]
        return torch.einsum(
            'ijd,df,ijf->ij',
            differences,
            self.mahalanobis_matrix,
            differences,
        )

    def _global_distances(self, labels):
        if self.global_distance == 'l2':
            return torch.cdist(labels, labels, p=2)
        return self._mahalanobis_distances(labels)

    def _anchor_temperatures(self, global_distances):
        n = global_distances.shape[0]
        off_diagonal = ~torch.eye(
            n, dtype=torch.bool, device=global_distances.device
        )
        mean_distance = global_distances[off_diagonal].view(n, n - 1).mean(1)
        return (
            self.temperature_alpha * mean_distance + self.temperature_beta
        ).detach()

    @staticmethod
    def _rnc_from_distances(
        features,
        relation_diffs,
        neighborhood_mask,
        temperatures,
    ):
        n = features.shape[0]
        logits = -torch.cdist(features, features, p=2) / temperatures[:, None]
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = logits.exp()

        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=features.device)
        logits = logits[off_diagonal].view(n, n - 1)
        exp_logits = exp_logits[off_diagonal].view(n, n - 1)
        relation_diffs = relation_diffs[off_diagonal].view(n, n - 1)
        neighborhood_mask = neighborhood_mask[off_diagonal].view(n, n - 1)

        sorted_diffs, order = relation_diffs.sort(dim=1)
        sorted_logits = logits.gather(1, order)
        sorted_exp_logits = exp_logits.gather(1, order)
        sorted_neighbors = neighborhood_mask.gather(1, order)

        suffix_sums = (
            sorted_exp_logits.masked_fill(~sorted_neighbors, 0)
            .flip(1).cumsum(1).flip(1)
        )

        group_starts = torch.cat([
            torch.ones((n, 1), dtype=torch.bool, device=features.device),
            sorted_diffs[:, 1:] != sorted_diffs[:, :-1],
        ], dim=1)
        positions = torch.arange(
            n - 1, device=features.device
        ).unsqueeze(0).expand_as(sorted_diffs)
        group_first = torch.where(
            group_starts, positions, torch.zeros_like(positions)
        ).cummax(dim=1).values
        denominator = suffix_sums.gather(1, group_first).clamp_min(
            torch.finfo(exp_logits.dtype).tiny
        )

        log_prob = sorted_logits - denominator.log()
        loss = -log_prob[sorted_neighbors].sum()

        return loss / neighborhood_mask.sum()

    def forward(self, features, labels, inputs=None):
        features, labels, inputs = self._expand_two_views(
            features, labels, inputs
        )
        n, output_dim = labels.shape
        if (
            self.global_distance == 'mahalanobis'
            and output_dim != self.mahalanobis_matrix.shape[0]
        ):
            raise ValueError(
                f'label dimension {output_dim} does not match fixed '
                f'Mahalanobis dimension {self.mahalanobis_matrix.shape[0]}.'
            )
        if n < 2:
            raise ValueError('Mo-RNC requires at least two samples per batch.')

        neighborhood_mask = self._input_neighborhood_mask(
            inputs, n, features.device
        )
        global_distances = self._global_distances(labels)
        temperatures = self._anchor_temperatures(global_distances)
        global_loss = self._rnc_from_distances(
            features,
            global_distances,
            neighborhood_mask,
            temperatures,
        )

        if self.local_weight == 0:
            self.last_global_loss = global_loss.detach()
            self.last_local_loss = global_loss.new_zeros(())
            self.last_temperature = temperatures.mean().detach()
            self.last_local_dims = torch.empty(
                0, dtype=torch.long, device='cpu'
            )
            return global_loss

        number_of_dims = min(self.local_dims, output_dim)
        selected_dims = torch.randperm(
            output_dim, device=labels.device
        )[:number_of_dims]
        local_loss = global_loss.new_zeros(())
        for dimension in selected_dims:
            values = labels[:, dimension:dimension + 1]
            local_distances = torch.cdist(values, values, p=1)
            local_loss = local_loss + self._rnc_from_distances(
                features,
                local_distances,
                neighborhood_mask,
                temperatures,
            )
        local_loss = local_loss / number_of_dims

        self.last_global_loss = global_loss.detach()
        self.last_local_loss = local_loss.detach()
        self.last_temperature = temperatures.mean().detach()
        self.last_local_dims = selected_dims.detach().cpu()
        return global_loss + self.local_weight * local_loss
