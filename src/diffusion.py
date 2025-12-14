### Troy / Chandhru

# Function 3: Diffusion takes in a set of points. 
#     - Takes in a map of points -> (Tangential Basis, Normal Baiss), fraction of noise to add along tangent, fraction to add along normal
#     - Noise function: If no basis is provided. Do standard method, If basis is provided, use the fractions to add noise along the bases provided.


#NOTE: Norm_Tangent_basis dictionary comes from (manifold_learning.py)
# calculate_tangent_and_normal(dataset) -> norm_tangent_basis = dict[datapoint, [tangent_basis, normal_basis]] 


# Diffusion(dataset, norm_tangent_basis, tangent_fraction, normal_fraction)
#     diffusion_train(num_iterations)
#     noiser(point, tangent_basis, normal_basis, tangent_fraction, normal_fraction) -> Noise? 

import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset,DataLoader
from src.manifold_learning import LocalFrames


class IndexedTensorDataset(Dataset):
    """
    Simple dataset that returns (x, index) so we can look up
    tangent/normal bases for each sample.
    """
    def __init__(self, X: torch.Tensor):
        """
        X : (N, D) float tensor
        """
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], idx



class DiffusionExperiment:

    def __init__(self,
                 X: np.ndarray,
                 local_frames: LocalFrames | None, # The tangent and normal basis are in here
                 tangent_fraction: float,
                 normal_fraction: float,
                 mixed_noise: bool = True, 
                 num_timesteps: int = 1000,
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        X : (N, D) numpy array of data points.
        local_frames : LocalFrames or None
            If provided, must correspond to the same X (same ordering).
        tangent_fraction, normal_fraction : floats
            RAW weights a0, b0 for how much noise to put in the tangent vs normal
            subspaces. These will be internally normalized so that
            E||noise||^2 ≈ D, where D is data_dim.
        """
        self.device = device
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.mixed_noise = mixed_noise
        self.tangent_fraction = tangent_fraction
        self.normal_fraction = normal_fraction
        
        self.device = device
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.mixed_noise = mixed_noise
        
        # Step 1: Setting up Data
        X = np.asarray(X, dtype=np.float32)
        self.X = torch.from_numpy(X).to(self.device)
        self.N, self.data_dim = self.X.shape
        
        self.dataset = IndexedTensorDataset(self.X)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True )
        
        # Step 2: Extract Manifold Frames
        self.has_frames = local_frames is not None and mixed_noise
        self.tangent_bases = None
        self.normal_bases = None
        
        if self.has_frames: 
            # Assume that local_frames.X is the same ordering as X used in here
            tangent = torch.from_numpy(local_frames.tangent.astype(np.float32))
            normal = torch.from_numpy(local_frames.normal.astype(np.float32))
            
            # Move to device (N, D, d_t), (N, D, d_n)
            self.tangent_bases = tangent.to(self.device)
            self.normal_bases = normal.to(self.device)
            
            # --- NEW: normalize tangent/normal fractions to fix total variance ---
            d_t = local_frames.intrinsic_dim           # tangent dimension
            D = self.data_dim
            d_n = D - d_t
            
            # Raw weights (a0, b0)
            a0 = float(tangent_fraction)
            b0 = float(normal_fraction)
            
            # Avoid degenerate case
            if a0 == 0.0 and b0 == 0.0:
                a0, b0 = 1.0, 1.0
            
            # E||noise||^2 = d_t * a^2 + d_n * b^2
            # We want this ≈ D (like N(0, I_D))
            denom = d_t * (a0 ** 2) + max(d_n, 0) * (b0 ** 2)
            if denom <= 1e-8:
                # Fallback: just use isotropic
                self.tangent_fraction = 1.0
                self.normal_fraction = 1.0
            else:
                target = D
                scale = math.sqrt(target / denom)
                self.tangent_fraction = a0 * scale
                self.normal_fraction = b0 * scale
        else:
            # No manifold information or not using mixed noise:
            # fall back to isotropic Gaussian in noiser().
            self.tangent_fraction = 1.0
            self.normal_fraction = 1.0
        
        # Step 3: Initialize Model and Optimizer
        self.model = DenoiserNetwork(data_dim=self.data_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Step 4: Beta schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def _batch_subspace_noise(self, bases):
        """
        Generates unit-variance Gaussian noise restricted to the subspace defined by 'bases'.
        
        Args:
            bases (torch.Tensor): Shape (B, D, k) where k is subspace dimension.
                                  Assumed to be orthonormal.
        Returns:
            torch.Tensor: Shape (B, D)
        """
        B, D, k = bases.shape
        if k == 0:
            return torch.zeros(B, D, device=self.device)
        # Sample z ~ N(0, I) in the subspace coefficient space
        z = torch.randn(B, k, 1, device=self.device)
        
        # Project back to ambient space: noise = Basis * z
        # (B, D, k) @ (B, k, 1) -> (B, D, 1)
        noise = torch.bmm(bases, z).squeeze(2)
        return noise

    def noiser(self, indices: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Generates noise based on the manifold geometry.
        
        Args:
            index (int): The index of the datapoint x0 so that the tangent
                and normal basis can be gathered from the dictionary
            d (int): The dimensionality of the data space
                               
        Returns:
            torch.Tensor: Generated noise batch.
        """
        # if has_frames then decompose noise into tanget and normal subspaces
        if self.has_frames:
            # Get bases for the batch
            # We need to transfer x0 to CPU for the dataset methods if they use numpy,
            # but our dataset supports tensor operations, so we keep it on device if possible.
            # However, dataset methods might re-create tensors, so let's just pass the tensors.
            
            # Look up the relevant tangent / normal bases: (B, D, d_t / d_n)
            tangent_bases = self.tangent_bases[indices]  # (B, D, d_t)
            normal_bases = self.normal_bases[indices]    # (B, D, d_n)
            
            # Generate independent noise components
            eps_tan = self._batch_subspace_noise(tangent_bases)
            eps_norm = self._batch_subspace_noise(normal_bases)
            
            # Combine
            # Note: In standard DDPM, isotropic noise is physically just (tangent_noise + normal_noise)
            # because the bases span the whole space.
            noise = self.tangent_fraction * eps_tan + self.normal_fraction * eps_norm
        # Otherwise generate noise following the typical diffusion proccess (no manifold info)
        else:
            noise = torch.randn_like(x0, device=self.device)
        
        return noise

    def diffusion_train(self, num_epochs: int):
        """Train for num_epochs over the dataset."""
        self.model.train()
        for epoch in range(1, num_epochs+1):
            epoch_loss = 0.0
            num_batches = 0

            for x0, idx in self.dataloader:
                x0 = x0.to(self.device)        # (B, D)
                idx = idx.to(self.device)      # (B,)
                B = x0.shape[0]
                
                # 1.  Sample time steps t ~ Uniform({0,...,T-1})
                t = torch.randint(
                    low=0,
                    high=self.num_timesteps,
                    size=(B,),
                    device=self.device,
                )                
                # 2. Generate manifold-aware or isotropic noise
                noise = self.noiser(idx, x0)   # (B, D)
                
                # 3. Add Noise (Forward diffusion: q(x_t | x_0))
                # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
                sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])[:, None]          # (B, 1)
                sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None]  # (B, 1)
                
                x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
                
                # 4. Predict Noise
                predicted_noise = self.model(x_t, t)   # (B, D)
                
                # 5. Optimization
                loss = self.criterion(predicted_noise, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

    def sample(self, dim=3, num_samples=1000):
        """
        Generates samples using the trained model via the reverse process.
        Returns:
            torch.Tensor: Generated samples (N, 3)
        """
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            # Note: Sampling usually starts from Isotropic Gaussian even if training was different,
            # though consistent testing might require thinking about this. 
            # For now, we use Standard Gaussian for x_T.
            x = torch.randn(num_samples, dim, device=self.device)
            
            for t_idx in reversed(range(self.num_timesteps)):
                t = torch.full((num_samples,), t_idx, device=self.device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.model(x, t)
                
                # Current alpha parameters
                beta_t = self.betas[t_idx]
                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alphas_cumprod[t_idx]
                
                # Helper term for mean calculation
                # mu = (1 / sqrt(alpha)) * (x - (beta / sqrt(1 - alpha_bar)) * eps)
                coeff = beta_t / torch.sqrt(1 - alpha_bar_t)
                mean = (1 / torch.sqrt(alpha_t)) * (x - coeff * predicted_noise)
                
                if t_idx > 0:
                    z = torch.randn_like(x)
                    sigma_t = torch.sqrt(beta_t)
                    x = mean + sigma_t * z
                else:
                    x = mean
                    
        return x

class DenoiserNetwork(nn.Module):
    """
    A simple MLP-based denoising network for 3D data.
    
    It accepts a noisy 3D coordinate and a time step, and outputs the predicted noise.
    This architecture is used for both standard DDPM and the Manifold-Specific noise experiments.
    """
    def __init__(self, data_dim=3, hidden_dim=256, time_dim=64, num_layers=4):
        """
        Args:
            data_dim (int): Dimension of the data (3 for Swiss Roll).
            hidden_dim (int): Number of units in hidden layers.
            time_dim (int): Dimension of the time embedding.
            num_layers (int): Number of hidden layers in the MLP.
        """
        super().__init__()
        
        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        # We concatenate data (3) + time_embedding (time_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Residual MLP Layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])
        
        # Final projection to data dimension (predicting noise)
        self.output_layer = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Noisy data batch of shape (Batch, data_dim)
            t (torch.Tensor): Time steps batch of shape (Batch,)
            
        Returns:
            torch.Tensor: Predicted noise of shape (Batch, data_dim)
        """
        # 1. Embed Time
        t_emb = self.time_mlp(t) # (Batch, time_dim)
        
        # 2. Concatenate Input and Time
        # x is (Batch, 3), t_emb is (Batch, 64) -> cat is (Batch, 67)
        x_input = torch.cat([x, t_emb], dim=1)
        
        # 3. Initial Features
        h = self.input_layer(x_input)
        
        # 4. Residual Layers
        for layer in self.layers:
            h = h + layer(h) # Residual connection
            
        # 5. Output Prediction
        return self.output_layer(h)
    
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal positional embeddings for time steps in diffusion models.
    Conceptually similar to Transformer position encodings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): Tensor of shape (batch_size,) containing time steps.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, dim) containing embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings