using TomoBOS
using Test
using Aqua
using JET

using StaticArrays
using LinearAlgebra
using OrderedCollections

@testset "TomoBOS.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TomoBOS)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(TomoBOS)
    end
    # Write your tests here.
end

include("helpers.jl")

@testset "project_point" begin
    T = Float64
    α = 10.0
    umax = 10
    vmax = 5
    u0, v0 = (umax+1)/2, (vmax+1)/2  # 画像中心 (ピクセル座標を1からumax, 1からvmaxの範囲とする)
    K = SMatrix{3,3,T,9}([α 0 u0; 0 α v0; 0 0 1])   # Intrinsic matrix
    Rc = SMatrix{3,3,T,9}(I)    # Rotation matrix for camera
    tc = SVector{3,T}(0.0, 0.0, 0.0) # Translation vector for camera
    cam = PinholeCamera{T}(Rc, tc, K, umax, vmax)

    Rb = SMatrix{3,3,T,9}(Ry(π)*Rz(π))    # Rotation matrix for board (180° around Y and Z)
    tb = SVector{3,T}(1.0, 0.0, 2.0) # Translation vector for board (2 units in front of camera)
    board = Board{T}(Rb, tb)

    """
    座標系の設定

        ᵇy
        ⊙───→ ᵇx
        │
        ↓ ᵇz


    ↑ ᶜz
    │
    ⊗───→ ᶜx (u)
    ᶜy (v)
    """

    ᵇx = SVector{3,T}(1.0, 2.0, 0.0)  # カメラ座標系では (2, -2, 2)

    # Test point in board coordinates (center of the board)
    u = project_point(ᵇx, cam, board)

    # 手計算すると u = (α*Xc/Zc + u0, α*Yc/Zc + v0) = (10*2/2+5.5, 10*(-2)/2+3) = (15.5, -7.0))
    @test u ≈ SVector{3,T}(15.5, -7.0, 1.0)

    bytes_allocated = @allocated project_point(ᵇx, cam, board)
    @test bytes_allocated == 0
end

@testset "normalize_points" begin
    T = Float64
    pts = [SVector{3,T}(1.0, 2.0, 1.0), SVector{3,T}(3.0, 4.0, 1.0), SVector{3,T}(5.0, 6.0, 1.0)]
    norm_pts, T_mat = TomoBOS.normalize_points(pts)

    n_pts = length(pts)

    # Check that the normalized points have zero mean
    μ1 = sum(p[1] for p in norm_pts) / n_pts
    μ2 = sum(p[2] for p in norm_pts) / n_pts
    @test μ1 ≈ 0.0
    @test μ2 ≈ 0.0

    # Check that the average distance from the origin is sqrt(2)
    mean_dist = sum(sqrt(p[1]^2 + p[2]^2) for p in norm_pts) / n_pts
    @test mean_dist ≈ sqrt(2)

    # Check that the normalization matrix is correct
    for i in 1:n_pts
        @test norm_pts[i] ≈ T_mat * pts[i]
    end
end

# @testset "estimate_initial_pose" begin
#     # Set up a synthetic problem with known camera and board poses, and synthetic marker data
#     (; cams_true, boards_true, all_marker_data) = create_circular_grid_setup()

#     # Estimate initial poses using the synthetic marker data
#     cams_init, boards_init = estimate_initial_pose(all_marker_data; ref_cam_id=1)

#     # Compare the estimated poses with the true poses
#     atol = 1e-5
#     for cam_id in keys(cams_true)
#         cam_true = cams_true[cam_id]
#         cam_init = cams_init[cam_id]
#         @test cam_init.R ≈ cam_true.R atol=atol
#         @test cam_init.t ≈ cam_true.t atol=atol
#     end

#     for board_id in keys(boards_true)
#         board_true = boards_true[board_id]
#         board_init = boards_init[board_id]
#         @test board_init.R ≈ board_true.R atol=atol
#         @test board_init.t ≈ board_true.t atol=atol
#     end
# end