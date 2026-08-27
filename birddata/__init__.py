"""bird_data — ETL and analysis of the Chicago bird-window-collision study.

Two entry points wrap the pipeline:

    bird-etl      fetch + clean + merge  -> data/processed/
    bird-analyze  correlations + figures/ + printed summary

or import the transformations directly:

    from birddata import etl, analyze
"""
__version__ = "0.1.0"

__all__ = ["etl", "analyze", "__version__"]
