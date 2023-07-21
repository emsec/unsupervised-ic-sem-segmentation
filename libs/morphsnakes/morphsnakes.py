# Adapted to OpenCL from https://github.com/pmneila/morphsnakes.git

import pyopencl as cl
import numpy as np

from numbers import Number


class MorphACWE:
    def __init__(self, cl_ctx, shape):
        """
        cl_ctx: OpenCL context from `cl.create_some_context()`
        """
        if cl_ctx is not None:
            self.cl_ctx = cl_ctx
        else:
            self.cl_ctx = cl.create_some_context(interactive=False)
        self.cl_queue = cl.CommandQueue(self.cl_ctx)
        with open("libs/morphsnakes/morphsnakes.cl") as source:
            self.cl_prog = cl.Program(self.cl_ctx, source.read()).build(options=["-cl-std=CL2.0"])

        self.shape = shape
        self.cl_image = cl.Image(self.cl_ctx, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8), shape)
        self.cl_u = cl.Image(self.cl_ctx, cl.mem_flags.READ_WRITE, cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8), shape)
        self.cl_u2 = cl.Image(self.cl_ctx, cl.mem_flags.READ_WRITE, cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8), shape)
        self.cl_pixel_sums = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS, size=24)


    def __call__(self, image, initial_threshold, iterations, smoothing=1, lambda1=1, lambda2=1):
        """Morphological Active Contours without Edges (MorphACWE)

        Active contours without edges implemented with morphological operators. It
        can be used to segment objects in images and volumes without well defined
        borders. It is required that the inside of the object looks different on
        average than the outside (i.e., the inner area of the object should be
        darker or lighter than the outer area on average).

        Parameters
        ----------
        image : (M, N) or (L, M, N) array
            Grayscale image or volume to be segmented.
        iterations : uint
            Number of iterations to run
        init_level_set : str, (M, N) array, or (L, M, N) array
            Initial level set. If an array is given, it will be binarized and used
            as the initial level set. If a string is given, it defines the method
            to generate a reasonable initial level set with the shape of the
            `image`. Accepted values are 'checkerboard' and 'circle'. See the
            documentation of `checkerboard_level_set` and `circle_level_set`
            respectively for details about how these level sets are created.
        smoothing : uint, optional
            Number of times the smoothing operator is applied per iteration.
            Reasonable values are around 1-4. Larger values lead to smoother
            segmentations.
        lambda1 : float, optional
            Weight parameter for the outer region. If `lambda1` is larger than
            `lambda2`, the outer region will contain a larger range of values than
            the inner region.
        lambda2 : float, optional
            Weight parameter for the inner region. If `lambda2` is larger than
            `lambda1`, the inner region will contain a larger range of values than
            the outer region.

        Returns
        -------
        out : (M, N) or (L, M, N) array
            Final segmentation (i.e., the final level set)

        See also
        --------
        circle_level_set, checkerboard_level_set

        Notes
        -----

        This is a version of the Chan-Vese algorithm that uses morphological
        operators instead of solving a partial differential equation (PDE) for the
        evolution of the contour. The set of morphological operators used in this
        algorithm are proved to be infinitesimally equivalent to the Chan-Vese PDE
        (see [1]_). However, morphological operators are do not suffer from the
        numerical stability issues typically found in PDEs (it is not necessary to
        find the right time step for the evolution), and are computationally
        faster.

        The algorithm and its theoretical derivation are described in [1]_.

        References
        ----------
        .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
            Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
            Transactions on Pattern Analysis and Machine Intelligence (PAMI),
            2014, DOI 10.1109/TPAMI.2013.106
        """

        #u = np.empty(image.shape, dtype=np.int8)
        #u_initial = np.empty(image.shape, dtype=np.int8)

        if len(image.shape) != 2:
            raise ValueError(f'Expected two dimensional image, got: {image.shape}')
        assert image.shape == self.shape[::-1]

        cl_threshold_mask = self.cl_prog.threshold_mask
        cl_init_pixel_sums = self.cl_prog.init_pixel_sums
        cl_compute_pixel_sums = self.cl_prog.compute_pixel_sums
        cl_advance_curve = self.cl_prog.advance_curve
        cl_erode = self.cl_prog.erode
        cl_dilate = self.cl_prog.dilate

        cl_init_pixel_sums.set_args(self.cl_pixel_sums)
        cl_compute_pixel_sums.set_args(self.cl_image, self.cl_u, self.cl_pixel_sums)
        cl_advance_curve.set_args(self.cl_image, self.cl_u, self.cl_u2, self.cl_pixel_sums, np.float32(lambda1), np.float32(lambda2))

        prev_ev = cl.enqueue_copy(self.cl_queue, dest=self.cl_image, src=image.astype(np.uint8), origin=(0, 0), region=self.shape, is_blocking=False)

        if isinstance(initial_threshold, Number):
            cl_threshold_mask.set_args(self.cl_image, self.cl_u, np.uint8(initial_threshold))
            prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_threshold_mask, self.shape, None, wait_for=[prev_ev])

        elif isinstance(initial_threshold, np.ndarray):
            assert initial_threshold.shape == self.shape[::-1]
            # Synchronous copy, simplifies event handling
            cl.enqueue_copy(self.cl_queue, dest=self.cl_u, src=initial_threshold.astype(np.uint8), origin=(0, 0), region=self.shape, is_blocking=True)

        else:
            raise ValueError(f'Expected int or numpy.ndarray as type for initial_threshold, got {initial_threshold.__class__}')

        smooth_op = True
        for _ in range(iterations):
            prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_init_pixel_sums, (1,), None)  # Run only one work item to init pixel_sums
            cl_compute_pixel_sums.set_arg(1, self.cl_u)
            prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_compute_pixel_sums, self.shape, (8, 8), wait_for=[prev_ev])
            cl_advance_curve.set_arg(1, self.cl_u)
            cl_advance_curve.set_arg(2, self.cl_u2)
            prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_advance_curve, self.shape, (8, 8), wait_for=[prev_ev])

            for _ in range(smoothing):
                if smooth_op:
                    cl_dilate.set_arg(0, self.cl_u2)
                    cl_dilate.set_arg(1, self.cl_u)
                    prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_dilate, self.shape, None, wait_for=[prev_ev])
                    cl_erode.set_arg(0, self.cl_u)
                    cl_erode.set_arg(1, self.cl_u2)
                    prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_erode, self.shape, None, wait_for=[prev_ev])
                else:
                    cl_erode.set_arg(0, self.cl_u2)
                    cl_erode.set_arg(1, self.cl_u)
                    prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_erode, self.shape, None, wait_for=[prev_ev])
                    cl_dilate.set_arg(0, self.cl_u)
                    cl_dilate.set_arg(1, self.cl_u2)
                    prev_ev = cl.enqueue_nd_range_kernel(self.cl_queue, cl_dilate, self.shape, None, wait_for=[prev_ev])
                smooth_op = not smooth_op

            self.cl_u, self.cl_u2 = self.cl_u2, self.cl_u

        u_out = np.empty_like(image, dtype=np.uint8)
        cl.enqueue_copy(self.cl_queue, dest=u_out, src=self.cl_u, origin=(0, 0), region=self.shape, wait_for=[prev_ev])
        return u_out




if __name__ == '__main__':
    from PIL import Image

    with open('dataset/sems/sem0000.png', "rb") as f:
            img = np.asarray(Image.open(f))

    morph = MorphACWE(cl.create_some_context(interactive=False), (4096, 3536))
    u_tracks = np.zeros_like(img)
    u_vias = np.zeros_like(img)
    u_tracks[img > 67] = 255
    u_vias[img > 157] = 255
    u_tracks = morph(img, u_tracks, 500, smoothing=3, lambda1=1, lambda2=2)
    u_vias = morph(img, u_vias, 500, smoothing=3, lambda1=2, lambda2=1)

    from matplotlib import pyplot as plt
    fig = plt.figure()

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img, cmap = 'gray', vmin=0, vmax=255), ax1.contour(u_tracks, [128], colors=['r']), ax1.contour(u_vias, [128], colors=['g'])
    ax1.set_title('Original'), ax1.set_xticks([]), ax1.set_yticks([])

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(u_tracks + u_vias, cmap = 'gray')
    ax2.set_title('Mask'), ax2.set_xticks([]), ax2.set_yticks([])

    plt.show()
