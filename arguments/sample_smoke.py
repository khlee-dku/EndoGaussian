# Minimal CUDA pipeline smoke test. This is not a reconstruction-quality config.
ModelParams = dict(
    extra_mark="endonerf",
    camera_extent=10,
    mode="monocular",
    no_fine=True,
)

OptimizationParams = dict(
    coarse_iterations=1,
    iterations=1,
    densify_until_iter=0,
    lambda_dssim=0,
    lambda_lpips=0,
)

ModelHiddenParams = dict(
    kplanes_config=dict(
        grid_dimensions=2,
        input_coordinate_dim=4,
        output_coordinate_dim=32,
        resolution=[32, 32, 32, 25],
    ),
    multires=[1, 2],
    defor_depth=0,
    net_width=32,
    plane_tv_weight=0,
    time_smoothness_weight=0,
    l1_time_planes=0,
)
