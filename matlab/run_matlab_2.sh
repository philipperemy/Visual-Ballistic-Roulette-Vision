#!/usr/bin/env bash
# argument is the name of the video.
/Applications/MATLAB_R2014b.app/bin/matlab -nojvm -r "try, compute_gradient_func('$1'); end; quit"
