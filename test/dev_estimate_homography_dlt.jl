using StaticArrays
using Random

using TomoBOS

using PythonCall
plt = pyimport("matplotlib.pyplot")

dir_save = joinpath(@__DIR__, "dev_estimate_homography_dlt")
if !isdir(dir_save)
    mkpath(dir_save)
end

T = Float64
rng = MersenneTwister(1234)

H_true = @SMatrix [2.0 0.3 -1.0; 0.1 1.5 2.0; 0.001 0.002 1.0]  # 適当なホモグラフィ行列 (回転行列だとシンプルすぎるので変更)
pts_src = [SVector{3,T}(rand(rng), rand(rng), 1.0) for _ in 1:10]
pts_dst = map(pts_src) do p
    p_trans = H_true * p
    SVector{3,T}(p_trans[1]/p_trans[3], p_trans[2]/p_trans[3], 1.0)  # Normalize to make the last coordinate 1
end

H_est = TomoBOS.estimate_homography_dlt(pts_src, pts_dst)
pts_dst_est = map(pts_src) do p
    p_trans = H_est * p
    SVector{3,T}(p_trans[1]/p_trans[3], p_trans[2]/p_trans[3], 1.0)  # Normalize to make the last coordinate 1
end

fig, ax = plt.subplots()
ax.plot([p[1] for p in pts_src], [p[2] for p in pts_src], ".", label="Source Points")
ax.plot([p[1] for p in pts_dst], [p[2] for p in pts_dst], ".", label="Destination Points")
ax.plot([p[1] for p in pts_dst_est], [p[2] for p in pts_dst_est], "x", label="Estimated Destination Points")
ax.set_aspect("equal")
ax.legend()

fig.tight_layout()
fig.savefig(joinpath(dir_save, "homography_estimation.png"), dpi=300)
# plt.show()