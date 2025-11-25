
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

