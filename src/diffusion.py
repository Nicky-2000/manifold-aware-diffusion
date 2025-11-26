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

class DiffusionExperiment:

    def __init__(self,
                 norm_tangent_basis,
                 tanget_fraction,
                 normal_fraction,
                 mixed_noise = True,
                 num_timesteps=1000, 
                 batch_size=128, 
                 lr=1e-3,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.norm_tangent_basis = norm_tangent_basis
        self.tanget_fraction = tanget_fraction
        self.normal_fraction = normal_fraction
        self.device = device
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.mixed_noise = mixed_noise
        
        # 1. Initialize Dataset and Model
        self.dataset = SwissRollDataset(num_samples=5000) #NOTE - refactor for generate_data.py
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        self.model = DenoiserNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 2. Define Beta Schedule (Linear schedule as in Ho et al.)
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
        # Sample z ~ N(0, I) in the subspace coefficient space
        z = torch.randn(B, k, 1, device=self.device)
        
        # Project back to ambient space: noise = Basis * z
        # (B, D, k) @ (B, k, 1) -> (B, D, 1)
        noise = torch.bmm(bases, z).squeeze(2)
        return noise

    def noiser(self, x0_index, d):
        """
        Generates noise based on the manifold geometry.
        
        Args:
            index (int): The index of the datapoint x0 so that the tangent
                and normal basis can be gathered from the dictionary
            d (int): The dimensionality of the data space
                               
        Returns:
            torch.Tensor: Generated noise batch.
        """
        # if mixed noise then decompose noise into tanget and normal subspaces
        if self.mixed_noise:
            # Get bases for the batch
            # We need to transfer x0 to CPU for the dataset methods if they use numpy,
            # but our dataset supports tensor operations, so we keep it on device if possible.
            # However, dataset methods might re-create tensors, so let's just pass the tensors.
            
            tangent_bases = self.norm_tangent_basis[x0_index]['tangent'] # (B, 3, 2)
            normal_bases = self.norm_tangent_basis[x0_index]['norm']   # (B, 3, 1)
            
            # Generate independent noise components
            eps_tan = self._batch_subspace_noise(tangent_bases)
            eps_norm = self._batch_subspace_noise(normal_bases)
            
            # Combine
            # Note: In standard DDPM, isotropic noise is physically just (tangent_noise + normal_noise)
            # because the bases span the whole space.
            noise = self.tanget_fraction * eps_tan + self.normal_fraction * eps_norm
        # Otherwise generate noise following the typical diffusion proccess (no manifold info)
        else:
            noise = torch.randn_like(d, device=self.device)
        
        return noise

    def diffusion_train(self, num_iterations):
        """Performs num_iterations epochs of training."""
        self.model.train()
        epoch_loss = 0
        for epoch in range(1, num_iterations+1):
            for x0 in self.dataloader: # NOTE this needs to be updated to get index for point
                x0 = x0.to(self.device)
                B = x0.shape[0]
                
                # 1. Sample Time Steps
                t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
                
                # 2. Generate Noise
                # This is where your custom logic happens!
                noise = self.get_mixed_noise(x0_index, x0.shape[-1]) #NOTE needs to be changed as above
                
                # 3. Add Noise (Forward Diffusion)
                # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
                sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])[:, None]
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
                
                x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
                
                # 4. Predict Noise
                predicted_noise = self.model(x_t, t)
                
                # 5. Optimization
                loss = self.criterion(predicted_noise, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            print(f'Loss for epoch {epoch}: {epoch_loss / len(self.dataloader)}')

    def sample(self, num_samples=1000):
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
            x = torch.randn(num_samples, 3, device=self.device)
            
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