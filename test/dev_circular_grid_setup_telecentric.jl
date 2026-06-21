using TomoBOS

using PythonCall
plt = pyimport("matplotlib.pyplot")

include("helpers.jl")

dir_save = joinpath(@__DIR__, "dev_circular_grid_setup_telecentric")
if !isdir(dir_save)
    mkpath(dir_save)
end

# Test setup for circular grid configuration
(; cams_true, boards_true, ᵇx_markers, all_marker_data) = create_circular_grid_setup(TelecentricCamera)

n_cams = length(cams_true)
n_boards = length(boards_true)


# Plot axes
function plot_axes(ax, origin, R, ii, jj; scale=0.05)
    x_axis = origin + R[:, 1] * scale
    y_axis = origin + R[:, 2] * scale
    z_axis = origin + R[:, 3] * scale

    ax.plot([origin[ii], x_axis[ii]], [origin[jj], x_axis[jj]], "r-")
    ax.plot([origin[ii], y_axis[ii]], [origin[jj], y_axis[jj]], "g-")
    ax.plot([origin[ii], z_axis[ii]], [origin[jj], z_axis[jj]], "b-")
end

# ==========================================
# Visualize the camera and board setup
# ==========================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ii, jj = 1, 3  # plot in the x-z plane

# Plot cameras
for (i, cam) in enumerate(values(cams_true))
    plot_axes(ax, cam.t, cam.R, ii, jj)
    ax.text(cam.t[ii], cam.t[jj], "Cam $(i)", fontsize="xx-small")
end

# Plot boards
for (i, board) in enumerate(values(boards_true))
    plot_axes(ax, board.t, board.R, ii, jj)
    ax.text(board.t[ii], board.t[jj], "Board $(i)", fontsize="xx-small")
    break  # Only plot the first board for clarity
end

fig.tight_layout()
fig.savefig(joinpath(dir_save, "camera_board_setup.png"), dpi=300)

# ==========================================
# Visualize the projected marker pixels
# ==========================================
fig, axs = plt.subplots(n_boards, n_cams, figsize=(8, 6))

for (i, board) in enumerate(values(boards_true))
    for (j, cam) in enumerate(values(cams_true))
        ax = axs[i-1, j-1]
        ax.set_xlim(1, cam.umax)
        ax.set_ylim(1, cam.vmax)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.tick_params(left=false, bottom=false, labelleft=false, labelbottom=false)
        ax.set_title("Board $(i), Camera $(j)", fontsize="xx-small")

        # Plot the projected marker corners (if any)
        mds = filter(md -> md.camera_id == j && md.board_id == i, all_marker_data)
        if isempty(mds)
            continue
        end
        marker_data = mds[1]
        u_markers = marker_data.u_markers
        if !isempty(u_markers)
            u_array = hcat(u_markers...)
            ax.plot(u_array[1, :], u_array[2, :], "r.", ms=5)
        end
    end
end

fig.tight_layout()
fig.savefig(joinpath(dir_save, "projected_markers.png"), dpi=300)
# plt.show()