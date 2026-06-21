using TomoBOS
using Test
using Aqua
using JET

using StaticArrays
using LinearAlgebra
using DataStructures

using Random

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

@testset "project_point (TelecentricCamera)" begin
    T = Float64
    mx = my = 10.0
    umax = 10
    vmax = 5
    cx, cy = (umax+1)/2, (vmax+1)/2  # 画像中心 (ピクセル座標を1からumax, 1からvmaxの範囲とする)
    Rc = SMatrix{3,3,T,9}(I)    # Rotation matrix for camera
    tc = SVector{3,T}(0.0, 0.0, 0.0) # Translation vector for camera
    cam = TelecentricCamera{T}(Rc, tc, mx, my, cx, cy, umax, vmax)

    Rb = SMatrix{3,3,T,9}(Ry(π)*Rz(π))    # Rotation matrix for board (180° around Y and Z)
    tb = SVector{3,T}(1.0, 0.0, 2.0) # Translation vector for board (2 units in front of camera)
    board = Board{T}(Rb, tb)

    ᵇx = SVector{3,T}(1.0, 2.0, 0.0)  # カメラ座標系では (2, -2, 2)

    # 手計算すると u = (mx*Xc + cx, my*Yc + cy) = (10*2+5.5, 10*(-2)+3) = (25.5, -17.0))
    u = project_point(ᵇx, cam, board)
    @test u ≈ SVector{3,T}(25.5, -17.0, 1.0)

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
    for i in eachindex(pts)
        @test norm_pts[i] ≈ T_mat * pts[i]
    end
end

@testset "estimate_homography_dlt" begin
    T = Float64
    rng = MersenneTwister(1234)

    H_true = @SMatrix [2.0 0.3 -1.0; 0.1 1.5 2.0; 0.001 0.002 1.0]  # 適当なホモグラフィ行列 (回転行列だとシンプルすぎるので変更)
    pts_src = [SVector{3,T}(rand(rng), rand(rng), 1.0) for _ in 1:10]
    pts_dst = map(pts_src) do p
        p_trans = H_true * p
        SVector{3,T}(p_trans[1]/p_trans[3], p_trans[2]/p_trans[3], 1.0)  # Normalize to make the last coordinate 1
    end

    H_est = TomoBOS.estimate_homography_dlt(pts_src, pts_dst)
    @test H_est ≈ H_true
end

@testset "estimate_single_board_pose" begin
    (; cams_true, boards_true, all_marker_data) = create_circular_grid_setup(PinholeCamera)

    cam_true = cams_true[1]
    board_true = boards_true[1]
    marker_data = filter(md -> md.camera_id == 1 && md.board_id == 1, all_marker_data)[1]  # 適当なマーカー観測データを選ぶ

    # Estimate the board pose using the marker data and the known camera pose
    cam_tmp = PinholeCamera{Float64}(SMatrix{3,3,Float64,9}(I), SVector{3,Float64}(0.0, 0.0, 0.0), cam_true.K, cam_true.umax, cam_true.vmax)
    R, t = TomoBOS.estimate_single_board_pose(marker_data, cam_tmp)

    # Compare the estimated board pose with the true board pose
    atol = 1e-8
    @test R ≈ board_true.R atol=atol
    @test t ≈ board_true.t atol=atol
end


@testset "estimate_initial_pose" begin
    # Set up a synthetic problem with known camera and board poses, and synthetic marker data
    (; cams_true, boards_true, all_marker_data) = create_circular_grid_setup(PinholeCamera)

    # Estimate initial poses using the synthetic marker data
    cams_known_param = SortedDict{Int, PinholeCamera{Float64}}([
            (camera_id, PinholeCamera{Float64}(SMatrix{3,3,Float64,9}(I), SVector{3,Float64}(0.0, 0.0, 0.0), cam.K, cam.umax, cam.vmax))
            for (camera_id, cam) in cams_true
        ])
    cams_init, boards_init = estimate_initial_poses(all_marker_data, cams_known_param; ref_camera_id=1)

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