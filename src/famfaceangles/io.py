"""Module containing some io functions"""
import re
import os
from mvpa2.support.nibabel import surf


def load_surface(input_fn, freesurfer_dir, which='pial',
                  fsaverage='fsaverage6'):
    """Automatically load pial surface given infile"""
    hemi = re.findall('hemi-([LR])', input_fn)[0]
    surf_fn = '{0}h.{1}.gii'.format(hemi.lower(), which)
    surf_fn = os.path.join(freesurfer_dir, fsaverage, 'SUMA', surf_fn)
    print("Loading {}".format(surf_fn))
    return surf.read(surf_fn)
