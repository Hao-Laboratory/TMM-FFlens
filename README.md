# TMM-FFlens

code of paper：

## Optimizing triangle mesh lenses for non-uniform illumination with an extended source

https://opg.optica.org/ol/abstract.cfm?URI=ol-48-7-1726

doi：10.1364/OL.485874

cite as:
~~~
@article{Li:23,
author = {Linpei Li and Xiang Hao},
journal = {Opt. Lett.},
keywords = {Freeform lenses; Freeform surfaces; Image metrics; Lens design; Nonimaging optics; Ray tracing},
number = {7},
pages = {1726--1729},
publisher = {Optica Publishing Group},
title = {Optimizing triangle mesh lenses for non-uniform illumination with an extended source},
volume = {48},
month = {Apr},
year = {2023},
url = {https://opg.optica.org/ol/abstract.cfm?URI=ol-48-7-1726},
doi = {10.1364/OL.485874},
abstract = {Precise control of irradiance distribution is a complicated problem for freeform lens design, especially when the target is non-uniform. Realistic sources are often simplified as zero-etendue ones in cases designed for content-rich irradiance fields while the surfaces are usually assumed smooth everywhere. These practices can limit the performance of the designs. We developed an efficient proxy for Monte Carlo (MC) ray tracing under extended sources, with the linear property of our triangle mesh (TM) freeform surface. Our designs show finer irradiance control compared to their counterparts from the LightTools design feature. One of the lenses is fabricated and evaluated in an experiment, and performed as expected.},
}
~~~


## usage
run trainer : optics_trainer.py

TensorBoard log file root: ./lightning_logs
