# Pages package for Demand Forecasting Dashboard
from .home import page_home
from .data_upload import page_data_upload
from .eda import page_eda
from .segmentation import page_segmentation_and_rules
from .outlier import page_outlier
from .forecasting import page_forecasting

__all__ = [
    'page_home',
    'page_data_upload',
    'page_eda',
    'page_segmentation_and_rules',
    'page_outlier',
    'page_forecasting'
]
