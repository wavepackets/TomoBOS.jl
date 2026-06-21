using Revise
using StaticArrays
using DataStructures

using Test

using TomoBOS

include("helpers.jl")

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