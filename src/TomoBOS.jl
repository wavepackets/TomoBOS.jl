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
    N = length(pts)

    # Ensure that the input points are not empty
    if N == 0
        throw(ArgumentError("Input points cannot be empty"))
    end

    # Compute the centroid of the points
    μ1 = sum(p[1] for p in pts) / N
    μ2 = sum(p[2] for p in pts) / N

    # Compute the average distance from the centroid
    mean_dist = sum(sqrt((p[1] - μ1)^2 + (p[2] - μ2)^2) for p in pts) / N

    # Ensure that the average distance is greater than zero to avoid division by zero
    if mean_dist ≈ 0.0
        throw(ArgumentError("All points are identical; cannot normalize"))
    end

    # Compute the scaling factor to make the average distance sqrt(2)
    scale = sqrt(2) / mean_dist

    # Construct the normalization matrix
    T_mat = @SMatrix [scale 0 -scale*μ1; 0 scale -scale*μ2; 0 0 1]

    # Normalize the points (memory allocation occurs here)
    norm_pts = map(p -> SVector{3,T}((p[1] - μ1) * scale, (p[2] - μ2) * scale, p[3]), pts)
    
    return norm_pts, T_mat
end

"""
    estimate_homography_dlt(pts_src::AbstractVector{<:SVector{3,T}}, pts_dst::AbstractVector{<:SVector{3,T}}) where {T<:Real}

Estimates a homography matrix using the Direct Linear Transform (DLT) algorithm given corresponding points in source and destination planes.
- `pts_src`: A vector of points in the source plane (SVector{3,T} where T<:Real)
- `pts_dst`: A vector of corresponding points in the destination plane (SVector{3,T} where T<:Real)
Returns the estimated homography matrix (3x3 SMatrix).

Note: At least 4 point correspondences are required to estimate the homography.
References: Hartley and Zisserman, "Multiple View Geometry in Computer Vision", 2004, p.109 (Algorithm 4.2)
"""
function estimate_homography_dlt(pts_src::AbstractVector{<:SVector{3,T}}, pts_dst::AbstractVector{<:SVector{3,T}}) where {T<:Real}
    N = length(pts_src)
    if N < 4
        throw(ArgumentError("At least 4 point correspondences are required to estimate homography"))
    end
    if length(pts_dst) != N
        throw(ArgumentError("Source and destination point sets must have the same number of points"))
    end

    # Normalize the source and destination points
    norm_src, T_src = normalize_points(pts_src)
    norm_dst, T_dst = normalize_points(pts_dst)

    # Construct the matrix A for DLT
    A = zeros(T, 2N, 9)
    for i in 1:N
        X = norm_src[i][1]
        Y = norm_src[i][2]
        x = norm_dst[i][1]
        y = norm_dst[i][2]

        A[2i-1, :] .= [X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x]
        A[2i, :]   .= [0, 0, 0, X, Y, 1, -y*X, -y*Y, -y]
    end

    # Perform SVD on A
    F = svd(A)
    h_norm = F.V[:, end]  # The homography is the last column of V

    # Reshape the homography vector into a 3x3 matrix
    H_norm = reshape(h_norm, (3, 3))'

    # Denormalize the homography
    H_unscaled = inv(T_dst) * H_norm * T_src
    H = H_unscaled / H_unscaled[3, 3]  # Normalize to make H[3,3] = 1
    return SMatrix{3,3,T,9}(H)
end

"""
    estimate_pose_from_homography(H::SMatrix{3,3,T,9}, K::SMatrix{3,3,T,9}) where T<:Real

Estimates the camera pose (rotation and translation) from a homography matrix and the camera intrinsic matrix.
- `H`: Homography matrix (3x3 SMatrix)
- `K`: Camera intrinsic matrix (3x3 SMatrix)
Returns a tuple of (R, t) where R is the rotation matrix and t is the translation vector.
"""
function estimate_pose_from_homography(H::SMatrix{3,3,T,9}, K::SMatrix{3,3,T,9}) where T<:Real
    invK = inv(K)
    r1_raw = invK * H[:, 1]
    r2_raw = invK * H[:, 2]
    t_raw = invK * H[:, 3]

    scale = (norm(r1_raw) + norm(r2_raw)) / 2  # norm(r1)だけでも良いが、r1とr2の両方を使うことでより安定する

    r1 = r1_raw / scale
    r2 = r2_raw / scale
    t = t_raw / scale

    r3 = cross(r1, r2)

    R_raw = @SMatrix [
        r1[1] r2[1] r3[1];
        r1[2] r2[2] r3[2];
        r1[3] r2[3] r3[3]
    ]

    # Ensure R is a proper rotation matrix using SVD
    F = svd(R_raw)
    R = F.U * F.Vt

    return R, t
end

"""
    estimate_single_board_pose(marker_data::MarkerData{T}, K::SMatrix{3,3,T,9}) where T<:Real

Estimates the pose of a single board given the marker data and camera intrinsic matrix.
- `marker_data`: MarkerData instance containing the detected marker corners and their corresponding board coordinates
- `K`: Camera intrinsic matrix (3x3 SMatrix)
Returns a tuple of (R, t) where R is the rotation matrix and t is the translation vector of the board with respect to the camera.
"""
function estimate_single_board_pose(marker_data::MarkerData{T}, K::SMatrix{3,3,T,9}) where T<:Real
    # Convert board coordinates to homogeneous coordinates
    ᵇx_markers_homo = [SVector{3,T}(x[1], x[2], 1.0) for x in marker_data.ᵇx_markers]

    H = estimate_homography_dlt(ᵇx_markers_homo, marker_data.u_markers)
    R, t = estimate_pose_from_homography(H, K)
    return R, t
end

end # module TomoBOS
