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

import numpy as np
# import scipy.constants as sci_const
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib

# ============================================================================
# LOGGING
# ============================================================================


log = logging.getLogger(__name__)
for hdlr in log.handlers[:]:
    log.removeHandler(hdlr)
log.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(asctime)s - %(name)s  - %(levelname)s ' +
                              '  %(message)s')


fh = logging.FileHandler('log.log', mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
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


# ============================================================================
# MAIN METHOD AND TESTING AREA
# ============================================================================


def main():
    """Description of main()"""


if __name__ == '__main__':
    main()
