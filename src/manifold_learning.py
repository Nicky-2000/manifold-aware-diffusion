

### Nicky 
# mainfold_approx.py
# Function 2 (manifold learning): Takes in a bunch of points. Returns a tangential basis and a normal basis for each point (ideally this is orthonormal)
# - If KNN - Takes in a number of neigbours to use

# calculate_tangent_and_normal(dataset) -> norm_tangent_basis = dict[datapoint, [tangent_basis, normal_basis]] 
#     - This could be input to the diffusion model so that we generate these things on the fly


# Manifold learning thing
# - We need to take in a data set. And then we need to output some representation of a manifold.


# Return Structure of the manifold function (for now... this could be very inefficient. 
# So it could be improved later )
# dict = {
#     0: {
#         norm: [[1,1,1,1], [2,2,2,2]]
#         tangent: [[2,2,2,5,6]]
#     }
#     1: {
#         norm: [[1,1,1,1], [2,2,2,2]]
#         tangent: [[2,2,2,5,6]]
#     }
# }