module TomoBOS

using StaticArrays
using LinearAlgebra

using PythonCall

# function test_opencv()
#     cv2 = pyimport("cv2")
#     println("OpenCV version: ", cv2.__version__)
# end

# ==========================================
# Exported types and functions to be used by other modules
# ==========================================
export PinholeCamera, Board, MarkerData
export project_point




# ==========================================
# Structs for cameras
# ==========================================
abstract type AbstractCamera end

"""
    PinholeCamera{T<:Real} <: AbstractCamera

A standard pinhole camera model using StaticArrays for high performance.
- `R`: Rotation matrix (3x3)
- `t`: Translation vector (3,)
- `K`: Intrinsic matrix (3x3)
- `umax`: Maximum u coordinate (image width)
- `vmax`: Maximum v coordinate (image height)
"""
struct PinholeCamera{T<:Real} <: AbstractCamera
    R::SMatrix{3,3,T,9}     # Rotation matrix
    t::SVector{3,T}         # Translation vector
    K::SMatrix{3,3,T,9}     # Intrinsic matrix
    umax::Int               # Maximum u coordinate
    vmax::Int               # Maximum v coordinate
end




# ==========================================
# Structs for boards
# ==========================================
abstract type AbstractBoard end

"""
    Board{T<:Real} <: AbstractBoard

A structure to represent a calibration board.
- `R`: Rotation matrix (3x3)
- `t`: Translation vector (3,)
"""
struct Board{T<:Real} <: AbstractBoard
    R::SMatrix{3,3,T,9}     # Rotation matrix
    t::SVector{3,T}         # Translation vector
end




# ==========================================
# Structs for marker data
# ==========================================
abstract type AbstractMarkerData end

"""
    MarkerData{T<:Real} <: AbstractMarkerData

A structure to hold detected marker data.
- `u_markers`: Pixel coordinates of marker corners (Vector of SVector{3,T}, assuming homogeneous coordinates (u, v, 1))
- `ᵇx_markers`: Board coordinates of marker corners (Vector of SVector{3,T})
- `camera_id`: ID of the camera that detected the marker
- `board_id`: ID of the board to which the marker belongs
"""
struct MarkerData{T<:Real} <: AbstractMarkerData
    u_markers::Vector{SVector{3,T}}  # Pixel coordinates of marker corners
    ᵇx_markers::Vector{SVector{3,T}}  # Board coordinates of marker corners
    camera_id::Int
    board_id::Int
end



# ==========================================
# Functions for camera calibration
# ==========================================
"""
    project_point(ᵇx::SVector{3,T}, cam::PinholeCamera{T}, board::Board{T}) where T<:Real

Projects a 3D point from board coordinates to pixel coordinates using the provided camera and board parameters.
- `ᵇx`: 3D point in board coordinates (SVector{3,T})
- `cam`: PinholeCamera instance
- `board`: Board instance
Returns a 3D point in pixel coordinates (SVector{3,T}).
"""
function project_point(ᵇx, cam::PinholeCamera{T}, board::Board{T}) where T<:Real
    # Transform point from board coordinates to global coordinates
    Rb, tb = board.R, board.t
    x = Rb * ᵇx .+ tb
    
    # Transform point from global coordinates to camera coordinates
    Rc, tc = cam.R, cam.t
    ᶜx = Rc' * (x .- tc)  # Assume x = Rc * ᶜx + tc => ᶜx = Rc' * (x - tc)

    # Project point onto image plane
    u = cam.K[1,1] * (ᶜx[1] / ᶜx[3]) + cam.K[1,3]
    v = cam.K[2,2] * (ᶜx[2] / ᶜx[3]) + cam.K[2,3]

    return SVector{3,T}(u, v, 1)
end

"""
    normalize_points(pts::AbstractVector{<:SVector{3,T}}) where {T<:Real}

Normalizes a set of 3D points and returns the normalized points along with the normalization matrix.
- `pts`: A vector of 3D points (SVector{3,T} where T<:Real)
Returns a tuple of (normalized points, normalization matrix).
"""
function normalize_points(pts::AbstractVector{<:SVector{3,T}}) where {T<:Real}
    M = length(pts)
    if M == 0
        return SVector{3,T}[], SMatrix{3,3,T,9}(I)
    end

    # Compute the centroid of the points
    μ1 = sum(p[1] for p in pts) / M
    μ2 = sum(p[2] for p in pts) / M

    # Compute the average distance from the centroid
    mean_dist = sum(sqrt((p[1] - μ1)^2 + (p[2] - μ2)^2) for p in pts) / M

    # Compute the scaling factor to make the average distance sqrt(2)
    scale = sqrt(2) / mean_dist

    # Construct the normalization matrix
    T_mat = @SMatrix [scale 0 -scale*μ1; 0 scale -scale*μ2; 0 0 1]

    # Normalize the points (memory allocation occurs here)
    norm_pts = map(p -> SVector{3,T}((p[1] - μ1) * scale, (p[2] - μ2) * scale, p[3]), pts)
    
    return norm_pts, T_mat
end



end # module TomoBOS
