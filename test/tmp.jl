using Revise
using StaticArrays
using DataStructures

using Test

using TomoBOS

include("helpers.jl")

@testset "estimate_single_board_pose" begin
    (; cams_true, boards_true, all_marker_data) = create_circular_grid_setup(TelecentricCamera)

    cam_true = cams_true[1]
    board_true = boards_true[1]
    marker_data = filter(md -> md.camera_id == 1 && md.board_id == 1, all_marker_data)[1]  # 適当なマーカー観測データを選ぶ

    # Estimate the board pose using the marker data and the known camera pose
    cam_tmp = TelecentricCamera{Float64}(
        SMatrix{3,3,Float64,9}(I),
        SVector{3,Float64}(0.0, 0.0, 0.0), 
        cam_true.mx, cam_true.my, cam_true.cx, cam_true.cy, 
        cam_true.umax, cam_true.vmax
        )
    poses = TomoBOS.estimate_single_board_pose(marker_data, cam_tmp)

    # Compare the estimated board pose with the true board pose
    atol = 1e-8

    R1, t1 = poses[1]  # 候補1
    R2, t2 = poses[2]  # 候補2

    @test xor(isapprox(R1, board_true.R, atol=atol), isapprox(R2, board_true.R, atol=atol))  # 候補1か候補2のどちらか一方のみが正しい (ボードが正面を向いていない場合)
    @test t1[1] ≈ board_true.t[1] atol=atol
    @test t1[2] ≈ board_true.t[2] atol=atol
    @test t1[3] ≈ 0.0 atol=atol    ## 奥行位置は不定なので、今の実装だと 0 ということにしてる

    @test t2[1] ≈ board_true.t[1] atol=atol
    @test t2[2] ≈ board_true.t[2] atol=atol
    @test t2[3] ≈ 0.0 atol=atol

    # display(board_true.R)
    # display(poses[1][1])
    # display(poses[2][1])

    # display(board_true.t)
    # display(t1)
    # display(t2)
end