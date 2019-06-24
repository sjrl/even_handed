"""
even_handed
Implementation of the even-handed subsystem selection for projection-based embedding.
"""

# Add imports here
from . import even_handed
from . import reaction_coord
from . import output_parser

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
