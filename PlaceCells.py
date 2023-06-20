import torch


class PlaceCells:
    def __init__(
        self,
        environment,
        npcs,
        pc_width=0.12,
        DoG=False,
        surround_scale=2,
        p=2.0,
        seed=0,
        dtype=torch.float32,
        **kwargs
    ):
        """
        Args:
                environment: ABCEnvironment class instance
                npcs: Number of place cells
                pc_width: Array-like or scalar. Tuning curve of place cells
        """
        self.environment = environment
        self.pcs = environment.sample_uniform(npcs, seed=seed)  # sample place cell centers (self.npcs, 2)
        self.pcs = torch.tensor(self.pcs, dtype=dtype)
        self.npcs = npcs
        self.pc_width = pc_width
        self.DoG = DoG
        self.surround_scale = surround_scale
        self.p = p
        self.seed = seed


    def softmax_response(self, pos):
        """
        Place cell response as modelled by Sorscher

        Parameters:
            pos (nsamples,2): cartesian positions to encode into place cell activities
        Returns:
            place cell activities (nsamples, self.npcs): place cell activities at given spatial positions
       """
        # distance to place cell center
        dists = torch.sum((pos[:, None] - self.pcs[None]) ** self.p, axis=-1) # (nsamples, self.npcs)
        activities = torch.nn.functional.softmax(
            -dists / (2 * self.pc_width ** 2), dim=-1
        )

        if self.DoG:
            activities -= torch.nn.functional.softmax(
                -dists / (2 * self.surround_scale * self.pc_width ** 2), dim=-1
            )

            # after DoG, activities is not a probability dist anymore
            # shift and rescale s.t it becomes a prob dist again.
            activities -= torch.min(
                activities, dim=-1, keepdim=True
            ).values  # returns idxs and values
            activities /= torch.sum(activities, dim=-1, keepdim=True)

        return activities

    def to_euclid(self, activities, k=3):
        """
        Decode place-cell activities to Euclidean coordinates - following Sorscher.
        OBS! This is an approximation to the actual Euclidean location,
        by considering the top k place-cell activities as if the agent is located
        at the average k place-cell center location
        """
        _, idxs = torch.topk(activities, k, dim=-1)
        return torch.mean(self.pcs[idxs], axis=-2)