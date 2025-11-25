### Troy / Chandhru

# Function 3: Diffusion takes in a set of points. 
#     - Takes in a map of points -> (Tangential Basis, Normal Baiss), fraction of noise to add along tangent, fraction to add along normal
#     - Noise function: If no basis is provided. Do standard method, If basis is provided, use the fractions to add noise along the bases provided.


#NOTE: Norm_Tangent_basis dictionary comes from (manifold_learning.py)
# calculate_tangent_and_normal(dataset) -> norm_tangent_basis = dict[datapoint, [tangent_basis, normal_basis]] 


# Diffusion(dataset, norm_tangent_basis, tangent_fraction, normal_fraction)
#     diffusion_train(num_iterations)
#     noiser(point, tangent_basis, normal_basis, tangent_fraction, normal_fraction) -> Noise? 