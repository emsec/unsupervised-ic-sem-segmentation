
typedef struct {
    atomic_ulong isum;
    atomic_ulong osum;
    atomic_uint icount;
    atomic_uint ocount;
} pixel_sums_t;

constant sampler_t clamp_to_edge = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

uchar4 read_neighbors4(read_only image2d_t u, const int2 coord) {
    return (uchar4) {
        read_imageui(u, clamp_to_edge, (int2) (coord.x - 1, coord.y)).x,  // left
        read_imageui(u, clamp_to_edge, (int2) (coord.x + 1, coord.y)).x,  // right
        read_imageui(u, clamp_to_edge, (int2) (coord.x, coord.y - 1)).x,  // top
        read_imageui(u, clamp_to_edge, (int2) (coord.x, coord.y + 1)).x,  // bottom
    };
}

uchar8 read_neighbors8(read_only image2d_t u, const int2 coord) {
    return (uchar8) {
        read_neighbors4(u, coord),
        read_imageui(u, clamp_to_edge, (int2) (coord.x - 1, coord.y - 1)).x,  // top left
        read_imageui(u, clamp_to_edge, (int2) (coord.x + 1, coord.y + 1)).x,  // bottom right
        read_imageui(u, clamp_to_edge, (int2) (coord.x + 1, coord.y - 1)).x,  // top right
        read_imageui(u, clamp_to_edge, (int2) (coord.x - 1, coord.y + 1)).x,  // bottom left
    };
}

kernel void threshold_mask(read_only image2d_t image, write_only image2d_t u, uchar threshold) {
    const int2 coord = {
        get_global_id(0),
        get_global_id(1),
    };

    const uchar img_px = read_imageui(image, coord).x;
    uchar u_px = img_px >= threshold;

    write_imageui(u, coord, (uint4)(u_px, 0, 0, 0));
}

kernel void init_pixel_sums(global pixel_sums_t* sums) {
    atomic_init(&sums->isum, 0);
    atomic_init(&sums->osum, 0);
    atomic_init(&sums->icount, 0);
    atomic_init(&sums->ocount, 0);
}

kernel void compute_pixel_sums(read_only image2d_t image, read_only image2d_t u, global pixel_sums_t* sums) {
    const int2 coord = {
        get_global_id(0),
        get_global_id(1),
    };

    const uchar img_px = read_imageui(image, coord).x;
    const uchar u_px = read_imageui(u, coord).x;

    ulong isum = u_px? img_px : 0;
    ulong osum = (!u_px)? img_px : 0;
    uint icount = u_px;
    uint ocount = !u_px;

    isum = work_group_reduce_add(isum);
    osum = work_group_reduce_add(osum);
    icount = work_group_reduce_add(icount);
    ocount = work_group_reduce_add(ocount);

    if (get_local_linear_id() == 0) {
        atomic_fetch_add_explicit(&sums->isum, isum, memory_order_relaxed);
        atomic_fetch_add_explicit(&sums->osum, osum, memory_order_relaxed);
        atomic_fetch_add_explicit(&sums->icount, icount, memory_order_relaxed);
        atomic_fetch_add_explicit(&sums->ocount, ocount, memory_order_relaxed);
    }
}

kernel void advance_curve(read_only image2d_t image, read_only image2d_t u, write_only image2d_t uout, global pixel_sums_t* sums, const float lambda1, const float lambda2) {
    const int2 coord = {
        get_global_id(0),
        get_global_id(1),
    };

    const uchar img_px = read_imageui(image, coord).x;

    uchar u_px = read_imageui(u, coord).x;
    const uchar4 u_neighbors = read_neighbors4(u, coord);

    ulong isum, osum;
    uint icount, ocount;
    if (get_local_linear_id() == 0) {
        isum = atomic_load_explicit(&sums->isum, memory_order_relaxed);
        osum = atomic_load_explicit(&sums->osum, memory_order_relaxed);
        icount = atomic_load_explicit(&sums->icount, memory_order_relaxed);
        ocount = atomic_load_explicit(&sums->ocount, memory_order_relaxed);
    }
    isum = work_group_broadcast(isum, 0, 0);
    osum = work_group_broadcast(osum, 0, 0);
    icount = work_group_broadcast(icount, 0, 0);
    ocount = work_group_broadcast(ocount, 0, 0);

    uchar du = abs(u_neighbors.s1 - u_neighbors.s0) + abs(u_neighbors.s3 - u_neighbors.s2);

    float c0 = osum / (ocount + FLT_MIN);
    float c1 = isum / (icount + FLT_MIN);
    float img_c0 = img_px - c0;
    float img_c1 = img_px - c1;

    float aux = du * ((lambda1 * img_c1 * img_c1) - (lambda2 * img_c0 * img_c0));
    u_px = (aux != 0)? (aux < 0) : u_px;

    write_imageui(uout, coord, (uint4)(u_px, 0, 0, 0));
}

kernel void erode(read_only image2d_t u, write_only image2d_t uout) {
    const int2 coord = {
        get_global_id(0),
        get_global_id(1),
    };
    char u_px = read_imageui(u, coord).x;
    const uchar8 n = read_neighbors8(u, coord);
    u_px = u_px && ((n.s0 && n.s1) || (n.s2 && n.s3) || (n.s4 && n.s5) || (n.s6 && n.s7));
    write_imageui(uout, coord, (uint4)(u_px, 0, 0, 0));
}

kernel void dilate(read_only image2d_t u, write_only image2d_t uout) {
    const int2 coord = {
        get_global_id(0),
        get_global_id(1),
    };
    char u_px = read_imageui(u, coord).x;
    const uchar8 n = read_neighbors8(u, coord);
    u_px = u_px || ((n.s0 || n.s1) && (n.s2 || n.s3) && (n.s4 || n.s5) && (n.s6 || n.s7));
    write_imageui(uout, coord, (uint4)(u_px, 0, 0, 0));
}
