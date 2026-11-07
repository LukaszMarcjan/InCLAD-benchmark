"""Per-method resumable state classes (UCAD, ReplayCAD, One-for-More).

Kept in a subpackage, and deliberately not imported from
``pyclad.resume.__init__``: these modules pull in the optional heavy
dependencies of the methods they wrap (timm for UCAD, diffusers for ReplayCAD
and One-for-More), and the infrastructure in ``pyclad.resume`` has to import
cleanly in an environment that has none of them.

Import the concrete classes from their own modules, e.g.::

    from pyclad.resume.methods.ucad import ResumableUCAD, ResumableUCADStrategy
"""
