using TomoBOS
using Test
using Aqua
using JET

using StaticArrays
using LinearAlgebra

@testset "TomoBOS.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TomoBOS)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(TomoBOS)
    end
    # Write your tests here.
end


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
    @test u ≈ SVector{2,T}(15.5, -7.0)

    bytes_allocated = @allocated project_point(ᵇx, cam, board)
    @test bytes_allocated == 0
end