# -*- coding: utf-8 -*-
"""
Created on %(date)s.

@author: James

summary

description

:REQUIRES:

:TODO:

"""

# Templates and markup notes

# >>SPYDER Note markers
#    #XXX: !
#    #TODO: ?
#    #FIXME: ?
#    #HINT: !
#    #TIP: !

# >>

# PDOCS: https://pdoc.dev/docs/pdoc.html,
#        https://pdoc3.github.io/pdoc/doc/pdoc/
# DOCSTRING CONVENTIONS: https://www.python.org/dev/peps/pep-0257/

# pdoc --html ./docs ./rate_calcs.py --force


# ============================================================================
# PROGRAM METADATA
# ============================================================================
__author__ = 'James Maldaner'
__contact__ = 'maldaner@ualberta.ca'
__copyright__ = ''
__license__ = ''
__date__ = '%(date)s'
__version__ = '0.1'
__all__ = []

# ============================================================================
# IMPORT STATEMENTS
# ============================================================================

import logging
import time
from functools import wraps
import re

from dataclasses import dataclass, field

from os import listdir  # to get list of file names in folder for looping


import numpy as np
import tifffile as tiff  # To load tiff files
# import scipy.constants as sci_const
# import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
# import matplotlib

# ============================================================================
# LOGGING
# ============================================================================


log = logging.getLogger(__name__)
for hdlr in log.handlers[:]:
    log.removeHandler(hdlr)
log.setLevel(logging.WARNING)


formatter = logging.Formatter('%(asctime)s - %(name)s  - %(levelname)s ' +
                              '  %(message)s')


fh = logging.FileHandler('log.log', mode='w')
fh.setLevel(logging.WARNING)
fh.setFormatter(formatter)
log.addHandler(fh)

sh = logging.StreamHandler()
sh.setLevel(logging.WARNING)
sh.setFormatter(formatter)
log.addHandler(sh)


def logging_dec(func):
    """Decorate functions with loging."""
    @wraps(func)
    def log_func(*args, **kwargs):
        start = time.perf_counter()  # Start the logging timer
        log_message = f'Running {func.__name__}. '  # First part of the log msg

        temp = func(*args, **kwargs)  # Run the wrapped function
        end = time.perf_counter()  # End the timer.
        log_message += f'[Exec time {end-start:.2E} s] '  # Last part of msg

        log.debug(log_message)  # log in debug
        return temp  # return the values from the wrapped function.
    log_func.__doc__ = func.__doc__  # Carry over the documentation.
    return log_func


# ============================================================================
# MATPLOTLIB SETTINGS
# ============================================================================

# matplotlib.rc_file(matplotlib.matplotlib_fname())
# plt.style.use('seaborn-colorblind')

# LaTeX rendering in plots
# plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
# plt.rc('text', usetex=True)

# ============================================================================
# REASIGNMENT STATEMENTS
# ============================================================================

pi = np.pi
exp = np.exp
sin = np.sin
cos = np.cos


SAVEFIGS = False


# ============================================================================
# MAIN METHOD AND TESTING AREA
# ============================================================================


@dataclass
class GaussianProfile:
    """Class to contain the data for gaussian profile and related methods."""

    axis: list
    """The axis values. Usually called x."""
    data: list
    """The data values. Usually called y."""
    gaus_dict: dict = field(init=False)
    """Dictionarry containing gaussian data."""
    radius: float = field(init=False)
    """1/e^2 beam radius."""

    def __post_init__(self):
        """Run as the object is being created."""
        log.debug('Gaussian post init')
        self.gaus_dict, self.radius = self.find_beam_radius()
    @logging_dec
    def find_beam_radius(self):
        """
        Find the FWHM for a function.

        #https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
        Inputs
        -----
        x: list or np.array
            This is the x data to be fit
        y: list or np.array
            This is the y data to be fit
        plot: Boolean (default: True)
            This chosses if the fit includes a plot.

        Output
        ------
        w0: float
            Beam 1/e^2 radius from the gaussian fit.
        """
        # Variable Formatting
        x_values = np.array(self.axis)  # Ensure the x data is a np array
        y_values = np.array(self.data)  # Ensure the y data is a np array

        # Initial guesses for the fitting parameters
        scale = max(y_values)  # The scaling is likely the maximum value

        # The mean is likely the weighted mean
        # The gaussian parameter is likely the following
        if sum(y_values) > 1:
            x_0 = sum(x_values*y_values)/sum(y_values)
            sigma = np.sqrt(sum(y_values * (x_values - x_0) ** 2) /
                            sum(y_values))
        else:
            sigma = 1e-6
            x_0 = 0

        # Fit the curve to the gaussian function with the default values
        try:
            popt, *_ = curve_fit(self.gaus, x_values, y_values,
                                 p0=[scale, x_0, sigma])
        except RuntimeError:
            popt = [1, 0, 10e-6]
        scale, x_0, sigma = popt  # Export the data
        radius = 2 * sigma  # The beam 1/e^2 radius that is defined as shown

        gaus_dict = {'scale': scale,
                     'x_0': x_0,
                     'sigma': sigma}
        return gaus_dict, radius

    def refine_beam_radius(self):
        """Perform Gaussian fit again in a smaller range."""
        x_values = np.array(self.axis)  # Ensure the x data is a np array
        y_values = np.array(self.data)  # Ensure the y data is a np array

        scale = self.gaus_dict['scale']
        x_0 = self.gaus_dict['x_0']
        sigma = self.gaus_dict['sigma']

        arg_small = np.argmin(np.abs(x_values - (x_0 - 4*sigma)))
        arg_big = np.argmin(np.abs(x_values - (x_0 + 4*sigma)))

        x_values = x_values[arg_small:arg_big + 1]
        y_values = y_values[arg_small:arg_big + 1]
        try:
            popt, *_ = curve_fit(self.gaus, x_values, y_values,
                                 p0=[scale, x_0, sigma])
        except TypeError:
            popt = [1, 0, 10e-6]
        scale, x_0, sigma = popt  # Export the data
        radius = 2 * sigma  # The beam 1/e^2 radius that is defined as shown

        gaus_dict = {'scale': scale,
                     'x_0': x_0,
                     'sigma': sigma}
        return gaus_dict, radius

    @staticmethod
    def gaus(x_value, scale, x_0, sigma):
        """Gaussian distrobution centered at x0, scalled by a, with a radius x.

        Inputs
        ------
        ***x***: *float*
            x argument

        ***a***: *float*
            Amplification of the gaussian pulse. Value of the maximum

        ***x0***: *float*
            x point where the gaussian is centered

        ***sigma***: *float*
            The variance of the gaussian

        Output
        ------
        *float*
            The gaussian value
        """
        return scale*exp(-(x_value-x_0)**2/(2*sigma**2))

    def imshow(self, parent_axes=False):
        """Show the gaussian profile data."""
        gaus_data = self.gaus(self.axis, **self.gaus_dict)

        x_min = self.gaus_dict['x_0'] - 4 * self.radius
        x_max = self.gaus_dict['x_0'] + 4 * self.radius

        if not parent_axes:
            _, axes = plt.subplots(1, 1)
        else:
            axes = parent_axes

        axes.plot(self.axis, self.data, 'bo', label='Data', alpha=0.3)
        axes.plot(self.axis, gaus_data, 'b', label='Fit')
        axes.hlines(self.gaus_dict['scale']*np.exp(-2),
                    x_min, x_max, 'r', label='1/e2 Reference')
        axes.vlines(x_min + self.radius*3, 0,
                    self.gaus_dict['scale'], 'r')
        axes.vlines(x_max - self.radius*3, 0,
                    self.gaus_dict['scale'], 'r')
        axes.set_xlabel('Position on sensor')
        axes.set_ylabel('Arb.')

        axes.legend()

        axes.set_xlim([x_min, x_max])

        if not parent_axes:
            plt.show()
            return None
        return axes



@dataclass
class ProfileImage:
    """Class to contain the camera images and related methods."""

    filename: str
    """File containing data."""
    save_folder: str = "../30-11-2022/30/Processed/I_SA_0A/"
    pixel_size: float = 5.2e-6
    """Width and height of a pixel."""
    max_offset = 50
    image: np.array = field(init=False)
    """Raw image data."""
    arg_max: list = field(init=False)
    """Arguments of the maximum loation."""

    def __post_init__(self):
        """Fill in remaining data for this class."""
        log.debug('Run post init')
        self.image = self._load_image()

        self.distance = self._get_distance_from_name()

        log.warning(f'Analysis of {self.distance*1e3:.0f} mm')
        self.arg_max = [self._find_max_x(self.image),
                        self._find_max_y(self.image)]

        self.find_max_scale()
        self.init_gaussian()

    def _load_image(self):
        """Load the data image that is a tiff."""
        image = tiff.imread(self.filename)
        return image

    def _get_distance_from_name(self):
        """
        Get distance from lens from the file name.

        Take the name of the file and extract the distance from the reference
        point in meters. Thisis done using regex and the data names should have
        the format (for example I_SA = 0, d = 50mm)
        I_SA_0A-d_50mm.tiff

        Parameters
        ----------
        ***file*** : *string*
        > The string containing data to be extracted.

        Returns
        -------
        *int*
        >The integer of the value.

        """
        if 'before' in self.filename:
            return 0
        list_values = re.findall('d_[0-9]+mm', self.filename)
        value = int(re.findall('[0-9]+', list_values[0])[0])
        return value * 1e-3

    @staticmethod
    def _find_max_y(image):
        """Find max in the y-direction.

        This is determined by the row with the larges sum.

        Parameters
        ----------
        ***image***: *np.array*
            Numpy array with the image data.

        Output
        ------
        ***image[y_max]***: *np.array*
            numpy array of the maximum row

        ***y_max***: *int*
            index of the max row

        """
        max_holder = 0
        y_max = 0
        for counter, i in enumerate(image):
            if sum(i) > max_holder:
                max_holder = sum(i)
                y_max = counter
        return y_max

    @staticmethod
    def _find_max_x(image):
        """Find max in the x direction.

        This function essentially transposes the image and calles the function
        to find the max in the y direction
        """
        return ProfileImage._find_max_y(np.transpose(image))

    def find_max_scale(self):
        """Iterate through slices around the initial beam center."""
        x_scale = []
        y_scale = []
        first_pass_x, _ = self.side_gaussian(0)

        max_offset = int(np.ceil(first_pass_x.radius/self.pixel_size))
        self.max_offset = max_offset
        offsets = np.arange(-max_offset, max_offset+1, 1)
        for offset in offsets:
            log.debug(f"Fitting for offset: {offset}")
            gauss_x, gauss_y = self.side_gaussian(offset)
            x_scale.append(gauss_x.gaus_dict['scale'])
            y_scale.append(gauss_y.gaus_dict['scale'])

        x_fit, x_radius = self.find_max_loc(offsets, x_scale)
        y_fit, y_radius = self.find_max_loc(offsets, y_scale)

        x_offset = int(np.round(x_fit['x_0']))
        y_offset = int(np.round(y_fit['x_0']))

        self.arg_max[1] += x_offset
        self.arg_max[0] += y_offset

        if sh.level < 30:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"Distance from lens = {self.distance:.3f} mm")
            for count_row, axs_row in enumerate(axes):
                for count_col, axs in enumerate(axs_row):
                    temp_offset = np.arange(-np.max((x_radius, y_radius))*2,
                                            np.max((x_radius, y_radius))*2)
                    if count_col == 0 and count_row == 0:
                        marker = '.'
                    elif count_col == 1 and count_row == 0:
                        marker = '+'
                    if count_row == 0:
                        axs.plot(offsets, x_scale, c='b', ls='', marker=marker,
                                 label='x scale')
                        axs.plot(offsets, y_scale, c='r', ls='', marker=marker,
                                 label='y scale')
                        axs.plot(temp_offset, self.gaus(temp_offset, **x_fit),
                                 c='b', label=f"x fit x0: {x_offset}")
                        axs.plot(temp_offset, self.gaus(temp_offset, **y_fit),
                                 c='r', label=f"y fit x0: {y_offset}")
                        axs.axvline(x_offset, 0, np.max(y_scale),
                                    ls='--', c='b', label='new x')
                        axs.axvline(y_offset, 0, np.max(y_scale),
                                    ls='--', c='r', label='new y')
                        axs.legend()
                        axs.set_xlabel('Pixel offset')
                        axs.set_ylabel('Arb.')
                        if count_col == 1 and count_row == 0:
                            axs.set_xlim([np.min(offsets), np.max(offsets)])
                            axs.set_ylim([np.min(np.concatenate((x_scale,
                                                                 y_scale))),
                                          np.max(np.concatenate((x_scale,
                                                                 y_scale)))])

            x_data = self.image[self.arg_max[1]]
            x_axis = self._get_camera_axis('x')
            y_data = np.transpose(self.image)[self.arg_max[0]]
            y_axis = self._get_camera_axis('y')

            axes[1, 0] = GaussianProfile(x_axis,
                                         x_data).imshow(parent_axes=axes[1, 0])
            axes[1, 1] = GaussianProfile(y_axis,
                                         y_data).imshow(parent_axes=axes[1, 1])

            axes[1, 0].set_title('x data')
            axes[1, 1].set_title('y data')
            filename = self.filename.split('/')[4].split('.')[0]
            plot_file = self.save_folder + filename

            if SAVEFIGS:
                plt.savefig(plot_file + '_gausfit.png')
            plt.show()

        log.debug("The correction to the x arg: " +
                  f"{offsets[np.argmax(x_scale)]}")
        log.debug("The correction to the y arg: " +
                  f"{offsets[np.argmax(y_scale)]}")



    def init_gaussian(self):
        """Init Gaussian profile for x and y."""
        log.debug('run gaussian init in profile image')
        x_data = self.image[self.arg_max[1]]
        x_axis = self._get_camera_axis('x')
        y_data = np.transpose(self.image)[self.arg_max[0]]
        y_axis = self._get_camera_axis('y')
        self.gauss_x = GaussianProfile(x_axis, x_data)
        self.gauss_y = GaussianProfile(y_axis, y_data)

    def side_gaussian(self, offset):
        """Find the gaussian of an offset from the max arg."""
        x_data = self.image[self.arg_max[1] + offset]
        x_axis = self._get_camera_axis('x')
        y_data = np.transpose(self.image)[self.arg_max[0] + offset]
        y_axis = self._get_camera_axis('y')
        gauss_x = GaussianProfile(x_axis, x_data)
        gauss_y = GaussianProfile(y_axis, y_data)
        log.debug(f"The scaling for offset = {offset} is" +
                  f"x:{gauss_x.gaus_dict['scale']} and " +
                  f"y: x:{gauss_x.gaus_dict['scale']} ")
        return gauss_x, gauss_y

    def _get_camera_axis(self, axis):
        """
        Return the camera axis with pixel size considered.

        Specify x or y and the function will return the desired axis

        Parameters
        ----------
        ***axis*** : *string*
            Either x or y and determins which axis to return

        Raises
        ------
        ***ValueError***
            If there is an issue with the input value.

        Returns
        -------
        ***axis*** : *np.array*
            A 1D array with step size equal to the pixel width.

        """
        if 'x' in axis and 'y' in axis:
            raise ValueError
        if 'y' in axis:
            axis = np.arange(0, np.min(np.shape(self.image))) * self.pixel_size
        elif 'x' in axis:
            axis = np.arange(0, np.max(np.shape(self.image))) * self.pixel_size
        else:
            raise ValueError
        return axis

    @staticmethod
    def find_max_loc(x_values, y_values):
        """
        Find the FWHM for a function.

        #https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
        Inputs
        -----
        x: list or np.array
            This is the x data to be fit
        y: list or np.array
            This is the y data to be fit
        plot: Boolean (default: True)
            This chosses if the fit includes a plot.

        Output
        ------
        w0: float
            Beam 1/e^2 radius from the gaussian fit.
        """
        # Variable Formatting
        x_values = np.array(x_values)  # Ensure the x data is a np array
        y_values = np.array(y_values)  # Ensure the y data is a np array

        # Initial guesses for the fitting parameters
        scale = max(y_values)  # The scaling is likely the maximum value

        # The mean is likely the weighted mean
        # The gaussian parameter is likely the following
        if sum(y_values) > 1:
            x_0 = sum(x_values*y_values)/sum(y_values)
            sigma = np.sqrt(sum(y_values * (x_values - x_0) ** 2) /
                            sum(y_values))
        else:
            sigma = 1e-6
            x_0 = 0

        # Fit the curve to the gaussian function with the default values
        try:
            popt, *_ = curve_fit(ProfileImage.gaus, x_values, y_values,
                                 p0=[scale, x_0, sigma])
        except RuntimeError:
            popt = [1, 0, 10e-6]
        scale, x_0, sigma = popt  # Export the data
        radius = 2 * sigma  # The beam 1/e^2 radius that is defined as shown

        gaus_dict = {'scale': scale,
                     'x_0': x_0,
                     'sigma': sigma}
        return gaus_dict, radius

    @staticmethod
    def gaus(x_value, scale, x_0, sigma):
        """Gaussian distrobution centered at x0, scalled by a, with a radius x.

        Inputs
        ------
        ***x***: *float*
            x argument

        ***a***: *float*
            Amplification of the gaussian pulse. Value of the maximum

        ***x0***: *float*
            x point where the gaussian is centered

        ***sigma***: *float*
            The variance of the gaussian

        Output
        ------
        *float*
            The gaussian value
        """
        return scale*exp(-(x_value-x_0)**2/(2*sigma**2))




@dataclass
class BeamRadius():
    """Class to contain data related to one beam configration."""

    current: int
    """The second amplifier current in amps."""
    date: str  # ex. 30-11-2022
    """The date of measurement with the form dd-mm-yyyy."""
    save_folder: str = "../30-11-2022/30/Processed/I_SA_0A/"
    w_0: tuple = field(init=False)
    z_0: tuple = field(init=False)
    lambda_0: float = 1040e-9
    z_r: tuple = field(init=False)
    w_radius_x: list = field(init=False)
    w_radius_y: list = field(init=False)
    d_lens: list = field(init=False)
    m_x: float = field(init=False)
    m_y: float = field(init=False)


    def __post_init__(self):
        """Init beam radius."""
        filelist = self.get_filelist()


        self.w_radius_x = []
        self.w_radius_y = []
        self.d_lens = []
        for file in filelist:
            distance, radius_x, radius_y = self.get_datapoints(self.folder() +
                                                               file)
            self.w_radius_x.append(radius_x)
            self.w_radius_y.append(radius_y)
            self.d_lens.append(distance)

        self.w_radius_x = np.array(self.w_radius_x[1:])
        self.w_radius_y = np.array(self.w_radius_y[1:])
        self.d_lens = np.array(self.d_lens[1:])
        self.find_w_0_and_z_0()
        self.z_r = tuple(self.get_z_r())
        self.fit_m2()
        self.imshow()
        if sh.level < 30:
            print(self)
    def __str__(self):
        """Return string representation."""
        string = f"Current: {self.current} A\n" + \
                 f"w0x: {self.w_0[0]*1e6:.2f} um\n" + \
                 f"w0y: {self.w_0[1]*1e6:.2f} um\n" + \
                 f"M2x: {self.m_x**2:.2f}\n" + \
                 f"Myx: {self.m_y**2:.2f}\n"
        return string

        def folder(self):
            """Return the folder of the data."""
            day = self.date.split('-')[0]
            return f"../{self.date}/{day}/I_SA_{self.current}A/"

    def folder(self):
        """Return the folder of the data."""
        day = self.date.split('-')[0]
        return f"../{self.date}/{day}/I_SA_{self.current}A/"

    def get_filelist(self):
        """Return the list of files in the data folder for iteration."""
        return list(listdir(self.folder()))

    def get_z_r(self):
        """Calculate rayleigh range."""
        log.debug('Running get z_r.')
        return pi * np.array(self.w_0)**2 / self.lambda_0

    def find_w_0_and_z_0(self):
        """Find the w0 and z0 from the min measured beam radius."""
        min_arg_x = np.argmin(self.w_radius_x)
        min_arg_y = np.argmin(self.w_radius_y)
        self.w_0 = (self.w_radius_x[min_arg_x], self.w_radius_y[min_arg_y])
        self.z_0 = (self.d_lens[min_arg_x], self.d_lens[min_arg_y])
        print(self.w_0)
        print(self.z_0)

    @staticmethod
    def w_from_m2(z_from_lens, w_0, m_not_squared, z_0, lambda0):
        """Exact relation between the beam waist and the M2.

        Inputs
        ------
        z: float
            distance from the reference point

        W0: float
            the 1/e^2 radius at the beam waist

        M: float
            The M from the Msquared meaure

        z0: float
            The beam waist location from the reference point

        lambda0: float (default 1040 e-9)
            Wavelength of the beam

        Output
        -----
        float
            The 1/e^2 radius of the beam at the specified point.
        """
        return np.sqrt(w_0**2 + m_not_squared ** 4 *
                       (lambda0 / (pi * w_0)) ** 2 * (z_from_lens - z_0)**2)

    @logging_dec
    def w_from_m2_for_fit_x(self, d_lens, m_not_squared):
        """Find the m2 for x."""
        return self.w_from_m2(d_lens, self.w_0[0], m_not_squared,
                              self.z_0[0], self.lambda_0)

    def w_from_m2_for_fit_y(self, d_lens, m_not_squared):
        """Find the m2 for y."""
        return self.w_from_m2(d_lens, self.w_0[1], m_not_squared,
                              self.z_0[1], self.lambda_0)

    def fit_m2(self):
        """Run both fit functions."""
        self.m_x = self.fit_m2_x(self.w_radius_x)
        self.m_y = self.fit_m2_y(self.w_radius_y)

    def fit_m2_x(self, w_radius):
        """Run fit for x."""
        m_not_squared = 1
        m_not_squared_lims = (1, 10)
        fit = curve_fit(self.w_from_m2_for_fit_x, self.d_lens, w_radius,
                        p0=[m_not_squared], bounds=m_not_squared_lims)
        return fit[0][0]

    def fit_m2_y(self, w_radius):
        """Run the fit for y."""
        m_not_squared = 1
        m_not_squared_lims = (1, 10)
        fit = curve_fit(self.w_from_m2_for_fit_y, self.d_lens, w_radius,
                        p0=[m_not_squared], bounds=m_not_squared_lims)
        return fit[0][0]

    @staticmethod
    def get_datapoints(filename):
        """Return data from beam image."""
        temp = ProfileImage(filename)
        return temp.distance, temp.gauss_x.radius, temp.gauss_y.radius

    def imshow(self):
        """Show beam image."""
        x_scale = 1e3
        y_scale = 1e6
        x_fit = np.linspace(np.min(self.d_lens), np.max(self.d_lens), 100)
        plt.figure(figsize=(10, 6))
        plt.plot(self.d_lens * x_scale, self.w_radius_x * y_scale,
                 'b.', label='x data')
        plt.plot(self.d_lens * x_scale, self.w_radius_y * y_scale,
                 'r.', label='y data')

        plot_file = self.save_folder + 'clean_beam_profile'
        if SAVEFIGS:
            plt.savefig(plot_file + '.png')
        plt.plot(x_fit * x_scale,
                 self.w_from_m2_for_fit_x(x_fit, self.m_x) * y_scale,
                 label='x fit', c='b')
        plt.plot(x_fit * x_scale,
                 self.w_from_m2_for_fit_y(x_fit, self.m_y) * y_scale,
                 label='y fit', c='r')

        y_max = np.max(np.concatenate((self.w_radius_x,
                                       self.w_radius_y)))*y_scale
        plt.axvline(self.z_0[0]*x_scale, 0, y_max,
                    label='x waist', c='b', ls='--')
        plt.axvline(self.z_0[1]*x_scale, 0, y_max,
                    label='y waist', c='r', ls='--')
        for z_r_scale in [1, 5]:
            l_s = 'dashdot'
            plt.axvline((self.z_0[0] + z_r_scale * self.z_r[0])*x_scale, 0,
                        y_max, c='b', ls=l_s, alpha=0.5)
            temp_x = plt.axvline((self.z_0[0] - z_r_scale * self.z_r[0]) *
                                 x_scale, 0, y_max, c='b', ls=l_s, alpha=0.5)
            plt.axvline((self.z_0[1] + z_r_scale * self.z_r[1])*x_scale, 0,
                        y_max, c='r', ls=l_s, alpha=0.5)
            temp_y = plt.axvline((self.z_0[1] - z_r_scale * self.z_r[1]) *
                                 x_scale, 0, y_max, c='r', ls=l_s, alpha=0.5)
        temp_y.set_label('y zr: 1 and 5')
        temp_x.set_label('x zr: 1 and 5')
        plt.xlabel('Distance from lens (mm)')
        plt.ylabel('Beam Radius (um)')
        plt.title(f"M2x: {self.m_x**2:.2f}, M2y: {self.m_y**2:.2f}")
        plt.legend()
        plot_file = self.save_folder + 'beam_profile'
        if SAVEFIGS:
            plt.savefig(plot_file + '.png')

        plt.show()



def main():
    """Description of main()"""
    print(BeamRadius(0,'30-11-2022'))


if __name__ == '__main__':
    main()
