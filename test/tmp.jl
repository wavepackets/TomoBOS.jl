using Revise
using StaticArrays
using DataStructures

using Test

using TomoBOS

include("helpers.jl")


@testset "estimate_initial_pose" begin
    # Set up a synthetic problem with known camera and board poses, and synthetic marker data
    (; cams_true, boards_true, all_marker_data) = create_circular_grid_setup()

    # Estimate initial poses using the synthetic marker data
    cam_params = SortedDict{Int, Any}([(camera_id, (; K=cam.K, umax=cam.umax, vmax=cam.vmax)) for (camera_id, cam) in cams_true])
    cams_init, boards_init = estimate_initial_poses(all_marker_data, cam_params; ref_camera_id=1)

    # Compare the estimated poses with the true poses
    atol = 1e-8
    for camera_id in keys(cams_true)
        cam_true = cams_true[camera_id]
        cam_init = cams_init[camera_id]
        @test cam_init.R ≈ cam_true.R atol=atol
        @test cam_init.t ≈ cam_true.t atol=atol
    end

    for board_id in keys(boards_true)
        board_true = boards_true[board_id]
        board_init = boards_init[board_id]
        @test board_init.R ≈ board_true.R atol=atol
        @test board_init.t ≈ board_true.t atol=atol
    end
end

# (; cams_true, boards_true, all_marker_data) = create_circular_grid_setup()

# # Estimate initial poses using the synthetic marker data
# cam_params = SortedDict{Int, Any}([(camera_id, (; K=cam.K, umax=cam.umax, vmax=cam.vmax)) for (camera_id, cam) in cams_true])


# struct PoseEdge{T}
#     to_node::Tuple{Symbol, Int}  # (:cam, id) or (:board, id)
#     ᶜRb::SMatrix{3,3,T,9}   # Rotation matrix from camera to board
#     ᶜtb::SVector{3,T}       # Translation vector from camera to board
# end

# T = Float64
# adj_dict = Dict{Tuple{Symbol, Int}, Vector{PoseEdge{T}}}()

# for marker_data in all_marker_data
#     cam_node = (:cam, marker_data.camera_id)
#     board_node = (:board, marker_data.board_id)

#     K = cam_params[marker_data.camera_id].K
#     ᶜRb, ᶜtb = TomoBOS.estimate_single_board_pose(marker_data, K)

#     # Add edges in both directions (camera to board and board to camera)
#     # `get!(adj_dict, cam_node, Vector{PoseEdge{T}}())` は「adj_dictにcam_nodeというキーが存在しない場合、空のVector{PoseEdge{T}}()を作成する」という意味
#     push!(get!(adj_dict, cam_node, Vector{PoseEdge{T}}()), PoseEdge(board_node, ᶜRb, ᶜtb))  # カメラ座標系でのボード姿勢を保存
#     push!(get!(adj_dict, board_node, Vector{PoseEdge{T}}()), PoseEdge(cam_node, ᶜRb, ᶜtb))  # この場合もカメラ座標系でのボード姿勢としておく
# end

# world_R = Dict{Tuple{Symbol, Int}, SMatrix{3,3,T,9}}()
# world_t = Dict{Tuple{Symbol, Int}, SVector{3,T}}()

# root_node = (:cam, 1)
# world_R[root_node] = SMatrix{3,3,T,9}(I)  # 基準カメラの姿勢をワールド座標系の原点とする
# world_t[root_node] = @SVector zeros(T, 3)

# queue = Queue{Tuple{Symbol, Int}}()
# enqueue!(queue, root_node)

# while !isempty(queue)
#     src = dequeue!(queue)

#     R_src = world_R[src]
#     t_src = world_t[src]

#     for edge in adj_dict[src]
#         (; ᶜRb, ᶜtb) = edge
#         dst = edge.to_node
#         if haskey(world_R, dst)
#             continue  # すでに姿勢が推定されているノードはスキップ
#         end

#         println((src, dst))

#         # 座標系を x = Rc * ᶜx + tc = Rb * ᵇx + tb と定義している
#         # ᶜx = ᶜRb * ᵇx + ᶜtb = Rc' * (x - tc) = Rc' * (Rb * ᵇx + tb - tc) = (Rc' * Rb) * ᵇx + (Rc' * (tb - tc))
#         # つまり ᶜRb = Rc' * Rb, ᶜtb = Rc' * (tb - tc) が成り立つ
#         # もしsrcがカメラ (Rc, tcが既知) なら、 Rb = Rc * ᶜRb, tb = Rc * ᶜtb + tc でボードの姿勢を推定できる
#         # もしsrcがボード (Rb, tbが既知) なら、 Rc = Rb * ᶜRb', tc = Rc * (-ᶜtb) + tb でカメラの姿勢を推定できる
#         if src[1] == :cam && dst[1] == :board
#             # カメラ Rc, tcが既知 -> ボードのRb, tbを推定
#             Rc, tc = R_src, t_src
#             Rb = Rc * ᶜRb
#             tb = Rc * ᶜtb + tc
#             R_dst, t_dst = Rb, tb
#         elseif src[1] == :board && dst[1] == :cam
#             # ボード Rb, tbが既知 -> カメラのRc, tcを推定
#             Rb, tb = R_src, t_src
#             Rc = Rb * ᶜRb'
#             tc = Rc * (- ᶜtb) + tb
#             R_dst, t_dst = Rc, tc
#         else
#             throw(ArgumentError("Invalid edge from $src to $dst"))
#         end

#         world_R[dst] = R_dst
#         world_t[dst] = t_dst

#         enqueue!(queue, dst)
#     end
# end

# # Construct the final camera and board dictionaries
# cams = SortedDict{Int, PinholeCamera{T}}()
# boards = SortedDict{Int, Board{T}}()

# for (node, R) in world_R
#     local t = world_t[node]
#     if node[1] == :cam
#         camera_id = node[2]
#         K = cam_params[camera_id].K
#         umax, vmax = cam_params[camera_id].umax, cam_params[camera_id].vmax
#         cams[camera_id] = PinholeCamera{T}(R, t, K, umax, vmax)
#     elseif node[1] == :board
#         board_id = node[2]
#         boards[board_id] = Board{T}(R, t)
#     end
# end

# cams_init = cams
# boards_init = boards

# atol = 1e-8
# for camera_id in keys(cams_true)
#     cam_true = cams_true[camera_id]
#     cam_init = cams_init[camera_id]
#     @test cam_init.R ≈ cam_true.R atol=atol
#     @test cam_init.t ≈ cam_true.t atol=atol
# end

# for board_id in keys(boards_true)
#     board_true = boards_true[board_id]
#     board_init = boards_init[board_id]
#     @test board_init.R ≈ board_true.R atol=atol
#     @test board_init.t ≈ board_true.t atol=atol
# end

