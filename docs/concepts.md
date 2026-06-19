# Concepts in Organelle Morphology

Here are explanations of some concepts used throughout Organelle Morphology.

## Project

Any analysis run starts by creating a Project.
It is defined by a directory where its outputs are stored in.
Multiple sources can be added to a project.

The contents of this directory are:

* Analysis results
* Caches (can be removed while OM is not running.)
* Logs

On the python side, the project is the main interaction point.

## Source

A source is a dataset stored as an `n5` file together with a `xml` file of the same name.
One source contains segmentations of one organelle type at different compression levels.

## Compression level

Always working with the data at the highest resolution is unpractical and slows everything down a lot.
The chosen compression level is set at the project level, applies therefor to all sources and all analysis.
