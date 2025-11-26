
##### Closed Form Manifold Function Known Metrics #####
# Known closed form equation for manifold:
#     - Metric: Generate a bunch of points using diffusion model. 
#         - Calculate the squared difference of each generated point from the closed form function for the swiss roll that was used. 

# known_function_metric(swiss_roll_function, generated_dataset) -> number


##### Closed Form Manifold Function UNKNOWN Metrics #####

# Unknown Closed form function metrics:
#     - Metric 1: Use the trained model to generate the same number of points that the original dataset was created
#         - Calculate sum of squared distances from diffusion created points to nearest neighbour in original dataset.
#         - Chamfer distance/metric


# Chamfer_metric(original_dataset, generated_dataset) -> number
import torch
#NOTE placeholder code with general logic, will need to be refactored
def compute_chamfer_distance(self, samples):
        """
        Computes the Chamfer Distance between generated samples and the true manifold.
        We approximate the true manifold by sampling fresh points from the dataset.
        """
        # Get ground truth samples
        true_samples = self.dataset.data.to(self.device) # (N_true, 3)
        gen_samples = samples.to(self.device)            # (N_gen, 3)
        
        # Pairwise distances
        # We process in chunks to avoid OOM if N is large
        
        # 1. For each gen sample, find closest true sample
        # Simple brute force for demo (optimized libs exist but this is portable)
        # Using a subset for speed if needed
        if len(true_samples) > 2000:
            true_samples = true_samples[:2000]
        if len(gen_samples) > 2000:
            gen_samples = gen_samples[:2000]
            
        # Expansion for broadcasting: (N_gen, 1, 3) - (1, N_true, 3)
        dists = torch.cdist(gen_samples, true_samples) # (N_gen, N_true) L2 distance
        
        dists_sq = dists ** 2
        
        min_dist_gen_to_true, _ = torch.min(dists_sq, dim=1)
        min_dist_true_to_gen, _ = torch.min(dists_sq, dim=0)
        
        chamfer_dist = torch.mean(min_dist_gen_to_true) + torch.mean(min_dist_true_to_gen)
        return chamfer_dist.item()