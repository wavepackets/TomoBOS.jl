using Revise
using StaticArrays
using DataStructures

using LinearAlgebra

using Test

using TomoBOS

include("helpers.jl")

using TomoBOS: estimate_single_board_pose

# Set up a synthetic problem with known camera and board poses, and synthetic marker data
(; cams_true, boards_true, all_marker_data) = create_circular_grid_setup(TelecentricCamera)

T = Float64
# Estimate initial poses using the synthetic marker data
cams_known_param = SortedDict{Int, TelecentricCamera{T}}([
        (camera_id, TelecentricCamera{T}(SMatrix{3,3,T,9}(I), SVector{3,T}(0.0, 0.0, 0.0), cam.mx, cam.my, cam.cx, cam.cy, cam.umax, cam.vmax))
        for (camera_id, cam) in cams_true
    ])
ref_cam_id = 1
default_tz = 0.5

# Step 1: 各カメラ・ボードペアのローカルな姿勢候補を計算
local_poses = Dict{Tuple{Int, Int}, Vector{Tuple{SMatrix{3,3,T,9}, SVector{3,T}}}}()  # (cam_id, board_id) => [(R1, t1), (R2, t2)]
board_to_cams = Dict{Int, Vector{Int}}()  # board_id => [cam_id1, cam_id2, ...] (どのカメラがどのボードを見ているか)
cam_to_boards = Dict{Int, Vector{Int}}()  # cam_id => [board_id1, board_id2, ...] (どのカメラがどのボードを見ているか)

for marker_data in all_marker_data
    cam_id = marker_data.cam_id
    board_id = marker_data.board_id

    cam = cams_known_param[cam_id]
    poses = estimate_single_board_pose(marker_data, cam; default_tz=default_tz)
    local_poses[(cam_id, board_id)] = poses

    push!(get!(board_to_cams, board_id, Vector{Int}()), cam_id)
    push!(get!(cam_to_boards, cam_id, Vector{Int}()), board_id)
end


# Step 2: 共通ボードを介して、カメラ間の相対回転を計算 (ボードの位置はStep 4で計算する)
cam_edges = Dict{Int, Vector{Tuple{Int, SMatrix{3,3,T,9}}}}()  # cam_id => [(neighbor_cam_id, R_rel), ...]
cam_ids = collect(keys(cams_known_param))
n_cams = length(cam_ids)

for i in 1:n_cams
    c1 = cam_ids[i]
    for j in (i+1):n_cams
        c2 = cam_ids[j]

        # 共通のボードを探す
        common_boards = intersect(get(cam_to_boards, c1, Int[]), get(cam_to_boards, c2, Int[]))
        if length(common_boards) < 2
            continue  # 共通のボードが2個以上無い場合はスキップ
        end

        b1 = common_boards[1]
        b2 = common_boards[2]

        # ボードb1から計算されるカメラc1から見たc2の相対回転 ᶜ¹Rc2 = ᶜ¹Rb1 * ᶜ²Rb1' (候補2×2=4通り)
        # ᶜ¹x = ᶜ¹Rb1 * ᵇ¹x + ᶜ¹tb1, ᶜ²x = ᶜ²Rb1 * ᵇ¹x + ᶜ²tb1 より
        # ᶜ¹x = ᶜ¹Rb1 * (ᶜ²Rb1' * (ᶜ²x - ᶜ²tb1)) + ᶜ¹tb1 = (ᶜ¹Rb1 * ᶜ²Rb1') * ᶜ²x + (ᶜ¹tb1 - ᶜ¹Rb1 * ᶜ²Rb1' * ᶜ²tb1) となるので、
        # よって ᶜ¹Rc2 = ᶜ¹Rb1 * ᶜ²Rb1' が得られる (ᶜ¹tc2 = ᶜ¹tb1 - ᶜ¹Rb1 * ᶜ²Rb1' * ᶜ²tb1 も得られる)。
        poses1_b1 = local_poses[(c1, b1)]
        poses2_b1 = local_poses[(c2, b1)]
        R12_b1 = [p1[1] * p2[1]' for p1 in poses1_b1, p2 in poses2_b1]  # ᶜ¹Rc2 = ᶜ¹Rb1 * ᶜ²Rb1' の候補

        # ボードb2から計算されるカメラc1から見たc2の相対回転 ᶜ¹Rc2 = ᶜ¹Rb2 * ᶜ²Rb2' (候補2×2=4通り)
        poses1_b2 = local_poses[(c1, b2)]
        poses2_b2 = local_poses[(c2, b2)]
        R12_b2 = [p1[1] * p2[1]' for p1 in poses1_b2, p2 in poses2_b2]  # ᶜ¹Rc2 = ᶜ¹Rb2 * ᶜ²Rb2' の候補

        # R12_b1 と R12_b2 の候補の中から、最も近い組み合わせを選ぶ
        min_d = Inf
        best_R12 = SMatrix{3,3,T,9}(I)
        for R_a in R12_b1
            for R_b in R12_b2
                d = norm(R_a - R_b)
                if d < min_d
                    min_d = d
                    best_R12 = R_a
                end
            end
        end

        F = svd(best_R12)
        R12_fixed = F.U * F.Vt  # 回転行列(直交かつdet=1)にする

        push!(get!(cam_edges, c1, Vector{Tuple{Int, SMatrix{3,3,T,9}}}()), (c2, R12_fixed))   # cam_edges[c1] に (c2, ᶜ¹Rc2) を追加
        push!(get!(cam_edges, c2, Vector{Tuple{Int, SMatrix{3,3,T,9}}}()), (c1, R12_fixed'))  # cam_edges[c2] に (c1, ᶜ²Rc1=ᶜ¹Rc2') を追加
    end
end

# Step 3: 基準カメラから出発して、カメラの姿勢をワールド座標系に統一
world_Rc = Dict{Int, SMatrix{3,3,T,9}}()
world_Rc[ref_cam_id] = SMatrix{3,3,T,9}(I)  # 基準カメラの姿勢をワールド座標系の原点とする

cam_queue = Queue{Int}()
push!(cam_queue, ref_cam_id)

while !isempty(cam_queue)
    src_cam = popfirst!(cam_queue)
    Rc1 = world_Rc[src_cam]

    for (dst_cam, R12) in get(cam_edges, src_cam, Vector{Tuple{Int, SMatrix{3,3,T,9}}}())
        if haskey(world_Rc, dst_cam)
            continue  # すでに姿勢が推定されているカメラはスキップ
        end

        # いま R12 は cam_edges[src_cam(=c1)] から取り出したから ᶜ¹Rc2
        # x = Rc1 * ᶜ¹x + tc1 = Rc1 * (ᶜ¹Rc2 * ᶜ²x + ᶜ¹tc2) + tc1 = (Rc1 * ᶜ¹Rc2) * ᶜ²x + (Rc1 * ᶜ¹tc2 + tc1)
        # これと x = Rc2 * ᶜ²x + tc2 を見比べれば、Rc2 = Rc1 * ᶜ¹Rc2 が得られる
        Rc2 = Rc1 * R12  
        world_Rc[dst_cam] = Rc2
        push!(cam_queue, dst_cam)
    end
end

# Step 4: 幅優先探索で、ボードの姿勢とカメラの位置を計算
Node_T = Tuple{Symbol, Int}

# 隣接リストは、接続関係のみ保持
# (:cam, cam_id) => [(:board, board_id), ...], (:board, board_id) => [(:cam, cam_id), ...]
adj_dict = Dict{Node_T, Vector{Node_T}}()
for marker_data in all_marker_data
    cam_node = (:cam, Int(marker_data.cam_id))
    board_node = (:board, Int(marker_data.board_id))
    push!(get!(adj_dict, cam_node, Vector{Node_T}()), board_node)
    push!(get!(adj_dict, board_node, Vector{Node_T}()), cam_node)
end

world_R = Dict{Node_T, SMatrix{3,3,T,9}}()
world_t = Dict{Node_T, SVector{3,T}}()

root_node = (:cam, ref_cam_id)
world_R[root_node] = SMatrix{3,3,T,9}(I)
world_t[root_node] = @SVector zeros(T, 3)

queue = Queue{Node_T}()
push!(queue, root_node)

while !isempty(queue)
    src = popfirst!(queue)
    R_src = world_R[src]
    t_src = world_t[src]

    for dst in get(adj_dict, src, Vector{Node_T}())
        if haskey(world_R, dst)
            continue  # すでに姿勢が推定されているノードはスキップ
        end

        if src[1] == :cam && dst[1] == :board
            cam_id, board_id = src[2], dst[2]
            Rc, tc = R_src, t_src  # 既に定まっている
            poses = local_poses[(cam_id, board_id)]
            
            # このボードを見ている他のカメラのうち、既に回転が推定されているものを選ぶ
            other_cams = filter(c -> c != cam_id && haskey(world_Rc, c), board_to_cams[board_id])

            best_pose_idx = 1
            if !isempty(other_cams)
                min_err = Inf
                for (idx, (ᶜRb_cand, _)) in enumerate(poses)  # 1 or 2 の候補
                    # ↑この候補を採用したとき、ワールド座標系では ↓ となる
                    Rb = Rc * ᶜRb_cand

                    # 他のカメラから見た姿勢に直した時に一番整合性がとれているものを探す
                    score = 0.0
                    for other_cam_id in other_cams
                        Rc_oc = world_Rc[other_cam_id]
                        ᶜRb_oc = Rc_oc' * Rb

                        # other_cam_idにも2つ候補があるので、小さい方を選ぶ
                        poses_other = local_poses[(other_cam_id, board_id)]
                        d1 = norm(poses_other[1][1] - ᶜRb_oc)
                        d2 = norm(poses_other[2][1] - ᶜRb_oc)
                        score += min(d1, d2)
                    end

                    if score < min_err
                        min_err = score
                        best_pose_idx = idx
                    end
                end
            end
        
            ᶜRb, ᶜtb = poses[best_pose_idx]
            Rb = Rc * ᶜRb
            tb = Rc * ᶜtb + tc
            world_R[dst] = Rb
            world_t[dst] = tb

        elseif src[1] == :board && dst[1] == :cam
            board_id, cam_id = src[2], dst[2]
            Rb, tb = R_src, t_src  # 既に定まっている
            poses = local_poses[(cam_id, board_id)]

            Rc = world_Rc[cam_id]  # 既に定まっている
            # ↑ これが pose 1 なのか pose 2 なのかを決める
            d1 = norm(poses[1][1] - Rc' * Rb)
            d2 = norm(poses[2][1] - Rc' * Rb)
            best_pose_idx = d1 < d2 ? 1 : 2

            (_, ᶜtb) = poses[best_pose_idx]
            tc = Rc * (- ᶜtb) + tb

            world_R[dst] = Rc
            world_t[dst] = tc

        else
            throw(ArgumentError("Invalid edge from $src to $dst"))
        end

        push!(queue, dst)
    end
end

# Step 5: Construct the final camera and board dictionaries
cams_out = SortedDict{Int, TelecentricCamera{T}}()
boards_out = SortedDict{Int, Board{T}}()

for (node, R) in world_R
    t = world_t[node]
    if node[1] == :cam
        cam_id = node[2]
        cam = cams_known_param[cam_id]
        cams_out[cam_id] = TelecentricCamera{T}(R, t, cam.mx, cam.my, cam.cx, cam.cy, cam.umax, cam.vmax)
    elseif node[1] == :board
        board_id = node[2]
        boards_out[board_id] = Board{T}(R, t)
    end
end



using PythonCall
plt = pyimport("matplotlib.pyplot")

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ii, jj = 1, 3  # plot in the x-z plane

# Plot cameras

function plot_axes(ax, origin, R, ii, jj; scale=0.05, alpha=1.0)
    x_axis = origin + R[:, 1] * scale
    y_axis = origin + R[:, 2] * scale
    z_axis = origin + R[:, 3] * scale

    ax.plot([origin[ii], x_axis[ii]], [origin[jj], x_axis[jj]], "r-", alpha=alpha)
    ax.plot([origin[ii], y_axis[ii]], [origin[jj], y_axis[jj]], "g-", alpha=alpha)
    ax.plot([origin[ii], z_axis[ii]], [origin[jj], z_axis[jj]], "b-", alpha=alpha)
end

for (i, cam) in enumerate(values(cams_true))
    plot_axes(ax, cam.t, cam.R, ii, jj; alpha=0.5)
    ax.text(cam.t[ii], cam.t[jj], "Cam $(i)", fontsize="xx-small")
end

for (i, cam) in enumerate(values(cams_out))
    plot_axes(ax, cam.t, cam.R, ii, jj; alpha=1.0)
    ax.text(cam.t[ii], cam.t[jj], "Cam $(i)", fontsize="xx-small")
end

# Plot boards
for (i, board) in enumerate(values(boards_true))
    plot_axes(ax, board.t, board.R, ii, jj; alpha=0.5)
    ax.text(board.t[ii], board.t[jj], "Board $(i)", fontsize="xx-small")
    break  # Only plot the first board for clarity
end

fig.tight_layout()
plt.show()