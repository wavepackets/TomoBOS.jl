module TomoBOS

using StaticArrays
using DataStructures

using LinearAlgebra

using PythonCall

# function test_opencv()
#     cv2 = pyimport("cv2")
#     println("OpenCV version: ", cv2.__version__)
# end

# ==========================================
# Exported types and functions to be used by other modules
# ==========================================
export PinholeCamera, TelecentricCamera
export Board, MarkerData
export project_point, estimate_initial_poses




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

struct TelecentricCamera{T<:Real} <: AbstractCamera
    R::SMatrix{3,3,T,9}    # Rotation matrix
    t::SVector{3,T}        # Translation vector
    mx::T                  # Magnification factor in x
    my::T                  # Magnification factor in y
    cx::T                  # Principal point x-coordinate
    cy::T                  # Principal point y-coordinate
    umax::Int              # Maximum u coordinate
    vmax::Int              # Maximum v coordinate
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

function project_point(ᵇx, cam::TelecentricCamera{T}, board::Board{T}) where T<:Real
    # Transform point from board coordinates to global coordinates
    Rb, tb = board.R, board.t
    x = Rb * ᵇx .+ tb
    
    # Transform point from global coordinates to camera coordinates
    Rc, tc = cam.R, cam.t
    ᶜx = Rc' * (x .- tc)  # Assume x = Rc * ᶜx + tc => ᶜx = Rc' * (x - tc)

    # Project point onto image plane using telecentric projection
    u = cam.mx * ᶜx[1] + cam.cx
    v = cam.my * ᶜx[2] + cam.cy

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
    estimate_single_board_pose(marker_data::MarkerData{T}, K::SMatrix{3,3,T,9}) where T<:Real

Estimates the pose of a single board given the marker data and camera intrinsic matrix.
- `marker_data`: MarkerData instance containing the detected marker corners and their corresponding board coordinates
- `cam`: PinholeCamera instance containing the camera intrinsic matrix (assume `K`, `umax`, and `vmax` are available in the camera parameters)
Returns a tuple of (R, t) where R is the rotation matrix and t is the translation vector of the board with respect to the camera.
"""
function estimate_single_board_pose(marker_data::MarkerData{T}, cam::PinholeCamera{T}) where T<:Real
    # Convert board coordinates to homogeneous coordinates
    ᵇx_markers_homo = [SVector{3,T}(x[1], x[2], 1.0) for x in marker_data.ᵇx_markers]

    H = estimate_homography_dlt(ᵇx_markers_homo, marker_data.u_markers)
    
    invK = inv(cam.K)
    r1_raw = invK * H[:, 1]
    r2_raw = invK * H[:, 2]
    t_raw = invK * H[:, 3]

    scale = (norm(r1_raw) + norm(r2_raw)) / 2  # norm(r1)だけでも良いが、r1とr2の両方を使うことでより安定する

    r1 = r1_raw / scale
    r2 = r2_raw / scale
    t = t_raw / scale

    r3 = cross(r1, r2)
    r3 /= norm(r3)  # Ensure r3 is a unit vector

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


function estimate_single_board_pose(marker_data::MarkerData{T}, cam::TelecentricCamera{T}) where T<:Real
    # Convert board coordinates to homogeneous coordinates
    ᵇx_markers_homo = [SVector{3,T}(x[1], x[2], 1.0) for x in marker_data.ᵇx_markers]

    H = estimate_homography_dlt(ᵇx_markers_homo, marker_data.u_markers)

    # Telecentric cameraの投影モデル (ref: Zhang & Chen (2026) https://doi.org/10.3390/s26051427)
    # u = mx * ᶜx[1] + cx
    # v = my * ᶜx[2] + cy
    # ᶜx[3] 成分の情報は消える
    #
    # ボード座標系からカメラ座標系への変換を ᶜx = ᶜRb ᵇx + ᶜtb とする (ᵇx[3] = 0 とする)
    # ただし成分を
    #       [r11 r12 r13]        [t1]
    # ᶜRb = [r21 r22 r23], ᶜtb = [t2]
    #       [r31 r32 r33]        [t3]
    # とすると、次のように書ける。
    # ᶜx[1] = r11*ᵇx[1] + r12*ᵇx[2] + t1 (ᵇx[3] = 0 のため)
    # ᶜx[2] = r21*ᵇx[1] + r22*ᵇx[2] + t2
    #
    # まとめると
    # [u]   [mx  0  cx] [ᶜx[1]]   [mx  0  cx] [r11  r12  t1] [ᵇx[1]]
    # [v] = [0  my  cy] [ᶜx[2]] = [0  my  cy] [r21  r22  t2] [ᵇx[2]]
    # [1]   [0   0   1] [  1  ]   [0   0   1] [  0    0   1] [  1  ]
    #                                  ↑ K         ↑ [Rs|ts; 0|1]
    #
    # 他方で、DLT法により以下のホモグラフィ行列が求まる。
    # [u]   [h11 h12 h13] [ᵇx[1]]
    # [v] = [h21 h22 h23] [ᵇx[2]]
    # [1]   [ 0   0   1 ] [  1  ]
    #             ↑ H
    #
    # 比較すれば H = K [Rs|ts] より K⁻¹ H = [Rs|ts; 0|1]、つまり
    # [1/mx   0   -cx/mx] [h11 h12 h13]   [h11/mx  h12/mx  (h13/mx - cx/mx)]   [r11 r12 t1]
    # [  0  1/my  -cy/my] [h21 h22 h23] = [h21/my  h22/my  (h23/my - cy/my)] = [r21 r22 t2]
    # [  0    0      1  ] [  0   0   1]   [   0       0            1       ]   [ 0   0   1]

    (; mx, my, cx, cy) = cam  # mx, myは既知とする (そうでない場合はオリジナルのZhang & Chenを参照)
    r11 = H[1, 1] / mx
    r12 = H[1, 2] / mx
    t1 = (H[1, 3] - cx) / mx

    r21 = H[2, 1] / my
    r22 = H[2, 2] / my
    t2 = (H[2, 3] - cy) / my

    t3 = 0.0  # 単一ホモグラフィからは一意に決まらないため、初期値として0を入れておく

    # ᶜRb は回転行列なので、r31² + r32² + r33² = 1 となるから、r31² = 1 - (r11² + r21²) が成り立つ。
    # また [r11 r21 r31] ⋅ [r12 r22 r32] = 0 であるから、r11*r12 + r21*r22 + r31*r32 = 0 であり、
    # つまり r31*r32 = -(r11*r12 + r21*r22) となるように、r31とr32の符号を決定する必要がある。
    r31_abs = sqrt(max(0.0, 1 - r11^2 - r21^2))   # ルート内が負にならないように max(0.0, ...) を使う
    r32_abs = sqrt(max(0.0, 1 - r12^2 - r22^2))

    product_val = -(r11*r12 + r21*r22)
    s = sign(product_val)  # r31とr32の符号を決定するための符号
    if s == 0
        s = 1.0  # 積が0の場合は正ということにする
    end

    poses = Vector{Tuple{SMatrix{3,3,T,9}, SVector{3,T}}}(undef, 2)

    # 候補1: r31 が正の場合
    let   # スコープを限定するために let ブロックを使う
        r31 = r31_abs
        r32 = s * r32_abs  # r31が正なら、r32の符号は s と一致

        r1 = SVector{3,T}(r11, r21, r31)  # 1列目
        r2 = SVector{3,T}(r12, r22, r32)  # 2列目
        r3 = cross(r1, r2)  # 3列目は外積で求める

        R_raw = @SMatrix [
            r1[1] r2[1] r3[1];
            r1[2] r2[2] r3[2];
            r1[3] r2[3] r3[3]
        ]

        F = svd(R_raw)
        R = F.U * F.Vt
        t = SVector{3,T}(t1, t2, t3)
        poses[1] = (R, t)
    end

    # 候補2: r31 が負の場合
    let
        r31 = -r31_abs
        r32 = -s * r32_abs  # r31が負なら、r32の符号は -s と一致

        r1 = SVector{3,T}(r11, r21, r31)  # 1列目
        r2 = SVector{3,T}(r12, r22, r32)  # 2列目
        r3 = cross(r1, r2)  # 3列目は外積で求める

        R_raw = @SMatrix [
            r1[1] r2[1] r3[1];
            r1[2] r2[2] r3[2];
            r1[3] r2[3] r3[3]
        ]

        F = svd(R_raw)
        R = F.U * F.Vt
        t = SVector{3,T}(t1, t2, t3)
        poses[2] = (R, t)
    end

    return poses  # 2つの候補姿勢を返す
end


struct PoseEdge{T, N}
    to_node::N  # (:cam, id) or (:board, id)
    ᶜRb::SMatrix{3,3,T,9}   # Rotation matrix from camera to board
    ᶜtb::SVector{3,T}       # Translation vector from camera to board
end

function estimate_initial_poses(
        all_marker_data::AbstractVector{<:MarkerData{T}}, 
        cams_known_param::AbstractDict{Int, <:PinholeCamera{T}}; 
        ref_camera_id=1
    ) where T<:Real

    # Construct a graph for the breadth-first search of camera and board poses
    # Example:
    # adj_list = Dict(
    #     (:cam, 1) => [PoseEdge((:board, 1), ᶜRb, ᶜtb), PoseEdge((:board, 2), ᶜRb, ᶜtb)],
    #     (:board, 1) => [PoseEdge((:cam, 1), ᶜRb, ᶜtb)],
    #     ...
    # )
    Node_T = Tuple{Symbol, Int}
    adj_dict = Dict{Node_T, Vector{PoseEdge{T, Node_T}}}()

    for marker_data in all_marker_data
        cam_node = (:cam, Int(marker_data.camera_id))
        board_node = (:board, Int(marker_data.board_id))

        cam = cams_known_param[marker_data.camera_id]
        ᶜRb, ᶜtb = estimate_single_board_pose(marker_data, cam)

        # Add edges in both directions (camera to board and board to camera)
        # `get!(adj_dict, cam_node, Vector{PoseEdge{T}}())` は「adj_dictにcam_nodeというキーが存在しない場合、空のVector{PoseEdge{T}}()を作成する」という意味
        push!(get!(adj_dict, cam_node, Vector{PoseEdge{T, Node_T}}()), PoseEdge{T, Node_T}(board_node, ᶜRb, ᶜtb))  # カメラ座標系でのボード姿勢を保存
        push!(get!(adj_dict, board_node, Vector{PoseEdge{T, Node_T}}()), PoseEdge{T, Node_T}(cam_node, ᶜRb, ᶜtb))  # この場合もカメラ座標系でのボード姿勢としておく
    end

    world_R = Dict{Node_T, SMatrix{3,3,T,9}}()
    world_t = Dict{Node_T, SVector{3,T}}()

    root_node = (:cam, ref_camera_id)
    world_R[root_node] = SMatrix{3,3,T,9}(I)  # 基準カメラの姿勢をワールド座標系の原点とする
    world_t[root_node] = @SVector zeros(T, 3)

    queue = Queue{Node_T}()
    push!(queue, root_node)

    while !isempty(queue)
        src = popfirst!(queue)

        R_src = world_R[src]
        t_src = world_t[src]

        for edge in adj_dict[src]
            (; ᶜRb, ᶜtb) = edge
            dst = edge.to_node
            if haskey(world_R, dst)
                continue  # すでに姿勢が推定されているノードはスキップ
            end

            # 座標系を x = Rc * ᶜx + tc = Rb * ᵇx + tb と定義する
            # ᶜx = ᶜRb * ᵇx + ᶜtb = Rc' * (x - tc) = Rc' * (Rb * ᵇx + tb - tc) = (Rc' * Rb) * ᵇx + (Rc' * (tb - tc))
            # つまり ᶜRb = Rc' * Rb, ᶜtb = Rc' * (tb - tc) が成り立つ
            # もしsrcがカメラ (Rc, tcが既知) なら、 Rb = Rc * ᶜRb, tb = Rc * ᶜtb + tc でボードの姿勢を推定できる
            # もしsrcがボード (Rb, tbが既知) なら、 Rc = Rb * ᶜRb', tc = Rc * (-ᶜtb) + tb でカメラの姿勢を推定できる
            if src[1] == :cam && dst[1] == :board
                # カメラ Rc, tcが既知 -> ボードのRb, tbを推定
                Rc, tc = R_src, t_src
                Rb = Rc * ᶜRb
                tb = Rc * ᶜtb + tc
                R_dst, t_dst = Rb, tb
            elseif src[1] == :board && dst[1] == :cam
                # ボード Rb, tbが既知 -> カメラのRc, tcを推定
                Rb, tb = R_src, t_src
                Rc = Rb * ᶜRb'
                tc = Rc * (- ᶜtb) + tb
                R_dst, t_dst = Rc, tc
            else
                throw(ArgumentError("Invalid edge from $src to $dst"))
            end

            world_R[dst] = R_dst
            world_t[dst] = t_dst

            push!(queue, dst)
        end
    end

    # Construct the final camera and board dictionaries
    cams_out = SortedDict{Int, PinholeCamera{T}}()
    boards_out = SortedDict{Int, Board{T}}()

    for (node, R) in world_R
        t = world_t[node]
        if node[1] == :cam
            camera_id = node[2]
            cam = cams_known_param[camera_id]
            cams_out[camera_id] = PinholeCamera{T}(R, t, cam.K, cam.umax, cam.vmax)
        elseif node[1] == :board
            board_id = node[2]
            boards_out[board_id] = Board{T}(R, t)
        end
    end

    return cams_out, boards_out
end

end # module TomoBOS
