using StaticArrays
using OrderedCollections
using LinearAlgebra

"""
Helper functions for testing TomoBOS.jl.
"""

Rx(θ) = [
    1.0 0.0 0.0;
    0.0 cos(θ) -sin(θ);
    0.0 sin(θ) cos(θ)
]
Ry(θ) = [
    cos(θ) 0.0 sin(θ);
    0.0 1.0 0.0;
    -sin(θ) 0.0 cos(θ)
]
Rz(θ) = [
    cos(θ) -sin(θ) 0.0;
    sin(θ) cos(θ) 0.0;
    0.0 0.0 1.0
]

function generate_synthetic_marker_data(cams, boards, ᵇx_markers)
    all_marker_data = Vector{MarkerData}()
    
    for (cam_id, cam) in cams
        for (board_id, board) in boards
            # もしボードが裏 or ほぼ真横を向いている(ボードのz軸(法線ベクトル)がカメラのz軸と同じ向き or ほぼゼロ)であれば、マーカーは見えないとする (全てのマーカーを無効とする)
            if dot(board.R[:,3], cam.R[:,3]) > -0.1
                continue
            end

            # マーカーのボード座標を、ピクセル座標に投影する
            u_markers = project_point.(ᵇx_markers, Ref(cam), Ref(board))

            # 投影されたマーカーのうち、画像内にあるものだけを有効とする
            valid_markers = [1<=u[1]<=cam.umax && 1<=u[2]<=cam.vmax for u in u_markers]
            
            # もし有効なマーカーが1つもなければ、次のボードに進む
            if !any(valid_markers)
                continue
            end

            marker_data = MarkerData(u_markers[valid_markers], ᵇx_markers[valid_markers], cam_id, board_id)
            push!(all_marker_data, marker_data)
        end
    end

    return all_marker_data
end

function create_circular_grid_setup()
    # Generate synthetic camera poses
    T = Float64
    α = 7500.0
    umax = 1440
    vmax = 1080
    u0, v0 = (umax+1)/2, (vmax+1)/2
    K = SMatrix{3,3,T,9}([α 0 u0; 0 α v0; 0 0 1])   # Intrinsic matrix

    n_cams = 8
    radius_cams = 0.5  # All cameras are placed on a circle of this radius around [0,0,radius_cams] in the world coordinate
    ᶜx = SVector{3,T}(0.0, 0.0, radius_cams)

    cams_true = OrderedDict{Int, PinholeCamera{T}}()
    Rc1 = SMatrix{3,3,T,9}(I)
    tc1 = SVector{3,T}(0.0, 0.0, 0.0)
    cams_true[1] = PinholeCamera{T}(Rc1, tc1, K, umax, vmax)

    for cam_id in 2:n_cams
        θ = π * (cam_id-1) / n_cams
        Rc = SMatrix{3,3,T,9}(Ry(θ))

        ## Assume Rc1 * ᶜx + tc1 = Rc * ᶜx + tc
        ## => tc = Rc1 * ᶜx - Rc * ᶜx + tc1
        tc = SVector{3,T}(
            Rc1 * ᶜx - Rc * ᶜx + tc1
        )
        cams_true[cam_id] = PinholeCamera{T}(Rc, tc, K, umax, vmax)
    end

    # Generate synthetic board poses
    n_boards = 5
    boards_true = OrderedDict{Int, Board{T}}()
    for board_id in 1:n_boards
        θ = π * (board_id-1) / n_boards
        Rb = SMatrix{3,3,T,9}(Ry(θ+π + π/4)*Rx(π/8))  # Boards are rotated to face the cameras, with a slight tilt
        tb = SVector{3,T}(0.0, 0.0, radius_cams)  # All boards are placed on the same circle as the cameras
        boards_true[board_id] = Board{T}(Rb, tb)
    end

    # Generate synthetic marker data
    ᵇx_markers = [SVector{3,T}(x, y, 0.0) for x in 0:0.02:0.1, y in 0:0.02:0.1]

    # Generate synthetic marker data for all cameras and boards
    all_marker_data = generate_synthetic_marker_data(cams_true, boards_true, ᵇx_markers)

    return (; cams_true, boards_true, ᵇx_markers, all_marker_data)
end
