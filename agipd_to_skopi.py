import numpy as np
import skopi as sk
from extra_geom import AGIPD_1MGeometry


def get_skopi_sensor(
                       input_detector: AGIPD_1MGeometry,
                       beam: sk.Beam = None,
                       single_pixel_height: float = 0.0002,
                       single_pixel_width: float = 0.0002,
                       simulate: bool = False,
                       particle: sk.Particle = None,
                       ) -> sk.UserDefinedDetector:
    """
    Generates and returns a skopi.UserDefinedDetector object from a
     extra_geom Geometry object. Requires the sensor pixel size,
     defaults to 0.2mm (AGIPD-1M).

    Parameters:
    input_detector: The input object.
    beam: The skopi.Beam object to use with the detector. Optional.
    pixel_height: The height of a single pixel, in meters.
    pixel_width: The width of a single pixel, in meters.
    simulate: Toggles a short simulation which shows the sensor visually.
     Defaults to False, requires a GPU with CUDA.
    particle: Only needed if simulate is True. The skopi.Particle object
     to use in the simulation. A basic particle can be found at
     https://github.com/chuckie82/skopi/blob/main/examples/input/pdb/2cex.pdb.
    """
    pixel_pos = input_detector.get_pixel_positions()
    pixel_matrix = input_detector.to_distortion_array()
    pixel_matrix_pixels = np.around(pixel_matrix / single_pixel_width, 1)

    p_center_x = pixel_pos[:, :, :, 0]  # only gets x coords
    p_center_y = pixel_pos[:, :, :, 1]  # only gets y coords

    p_map = pixel_matrix_pixels.mean(axis=2)[:, :, 1:] \
        .reshape(pixel_pos.shape[0], pixel_pos.shape[1], pixel_pos.shape[2], 2)

    pixel_height_array = single_pixel_height * np.ones(p_center_y.shape)
    pixel_width_array = single_pixel_width * np.ones(p_center_x.shape)

    detector_geometry = {
        'panel number': pixel_pos.shape[0],
        'panel pixel num x': pixel_pos.shape[1],
        'panel pixel num y': pixel_pos.shape[2],
        'detector distance': 0.3,  # distance between detector and sample in m
        'pixel width': pixel_width_array,  # width of each pixel as array
        'pixel height': pixel_height_array,  # height of each pixel as array
        'pixel center x': p_center_x,  # x-coordinate of each pixel center
        'pixel center y': p_center_y,  # y-coordinate of each pixel center
        'pixel map': p_map,  # map to assemble detector
    }
    if not simulate:
        return sk.UserDefinedDetector(geom=detector_geometry, beam=beam)
    else:
        import matplotlib.pyplot as plt
        import matplotlib
        detector = sk.UserDefinedDetector(geom=detector_geometry, beam=beam)
        experiment = sk.SPIExperiment(detector, beam, particle)
        dataset = np.zeros((1,) + detector.shape, np.float32)
        beam = sk.Beam(photon_energy=4600, fluence=1e12, focus_radius=1e-7)
        photons = np.zeros((1,) + detector.shape, np.int32)
        dataset[0] = experiment.generate_image_stack(return_intensities=True)
        N = dataset.shape[0]
        plt.figure(figsize=(20, 15/N+1))
        for i in range(N):
            plt.subplot(1, N, i+1)
            img = experiment.det.assemble_image_stack(dataset[i])
            plt.imshow(img, norm=matplotlib.colors.LogNorm())
        plt.show()
        return detector


sample_agipd = AGIPD_1MGeometry.from_quad_positions(quad_pos=[(-525, 625),
                                                              (-550, -10),
                                                              (520, -160),
                                                              (542.5, 475)])
sample_beam = sk.Beam(photon_energy=4600, fluence=1e12, focus_radius=1e-7)
sample_particle = sk.Particle()
sample_particle.read_pdb('2CEX.pdb', ff='WK')
get_skopi_sensor(input_detector=sample_agipd,
                 single_pixel_width=0.0002,
                 single_pixel_height=0.0002,
                 simulate=True,
                 beam=sample_beam,
                 particle=sample_particle)
